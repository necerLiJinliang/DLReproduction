import heapq
import json
from typing import Optional
from torch.utils.data import Dataset
import json
import torch
import re
import jieba
import random
from tqdm import tqdm
import numpy as np


class HuffmanNode:
    def __init__(
        self,
        char: Optional[str],
        freq: int,
    ) -> None:
        self.char = char
        self.freq = freq
        self.right = None
        self.left = None

    # 最小堆比较规则
    def __lt__(self, other):
        return self.freq < other.freq


def construct_Huffman_tree(
    words_freq: dict,
):
    huffman_heap = [HuffmanNode(key, val) for key, val in words_freq.items()]
    heapq.heapify(huffman_heap)

    while len(huffman_heap) > 1:
        # 取出两个频率最小的节点
        node1 = heapq.heappop(huffman_heap)
        node2 = heapq.heappop(huffman_heap)
        # 构造新的节点，频率为两个节点之和
        node_comb = HuffmanNode(None, node1.freq + node2.freq)
        node_comb.left = node1
        node_comb.right = node2
        heapq.heappush(huffman_heap, node_comb)
    return huffman_heap[0]


def get_Huffman_codes(
    huffman_node: HuffmanNode,
    current_code: str,
    codes: dict,
    no_leaf_codes: set,
):
    # 非叶子节点
    if huffman_node.char is not None:
        codes[huffman_node.char] = current_code
        # return
    else:
        no_leaf_codes.add(current_code)
    # 左0
    if huffman_node.left:
        get_Huffman_codes(huffman_node.left, current_code + "0", codes, no_leaf_codes)
    # 右1
    if huffman_node.right:
        get_Huffman_codes(huffman_node.right, current_code + "1", codes, no_leaf_codes)

    return codes, no_leaf_codes


def get_path_node_ids(huffman_code: str, no_leaf_code2index: dict) -> list:
    """根据哈夫曼编码返回经过的节点的编号

    Args:
        huffman_code (list): 哈夫曼编码

    Returns:
        list: 经过的节点的编号

    Explanation:
        哈夫曼是前缀编码，可以依次根据前缀确定经过了哪些节点
    """
    node_ids = []
    for i in range(len(huffman_code)):
        node_ids.append(no_leaf_code2index[huffman_code[:i]])
    return node_ids


class CBOWDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        word2id_file: str,
        huffman_code_file: str,
        no_leaf_code2index_file: str,
        window_size: int = 5,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        with open(file_path, "r") as f:
            self.sentences = json.load(f)
        self.word2id = json.load(open(word2id_file))
        self.id2word = {val: key for key, val in self.word2id.items()}
        self.huffman_code = json.load(open(huffman_code_file))
        self.huffman_code = {
            self.word2id[key]: val for key, val in self.huffman_code.items()
        }
        # self.huffman_code = [[int(t) for t in code] for code in self.huffman_code]
        self.no_leaf_code2index = json.load(open(no_leaf_code2index_file))
        self.words_num = len(self.word2id)
        ## 根据滑动窗口构造数据
        # self.data = self.data_process(self.sentences)
        self.data = self._data_process(self.sentences)

    def _data_process(self, sentences):
        def remove_punctuation(text):
            text = re.sub(r"[^\w\s]", "", text)  # 去除标点符号
            return text

        sentences = [s for s in sentences if len(s) >= self.window_size]
        data = []
        for sentence in sentences:
            for i in range(len(sentence) - self.window_size + 1):
                t = sentence[i : i + self.window_size]
                t = [self.word2id[s] for s in t]
                data.append(t)

        return data

    def __getitem__(self, index):
        item = self.data[index].copy()
        target_id = item.pop(self.window_size // 2)  # item 将中心点去掉了
        input_vector = torch.zeros(self.words_num)
        input_vector[item] = 1
        path_nodes_index = get_path_node_ids(
            self.huffman_code[target_id], self.no_leaf_code2index
        )
        huffman_code = self.huffman_code[target_id]
        return {
            "input_vector": input_vector,
            "path_nodes_index": path_nodes_index,
            "huffman_code": huffman_code,
        }

    def __len__(self):
        return len(self.data)

    def collate_fn(batch):
        inputs_vector = [f["input_vector"] for f in batch]
        huffman_codes = [[int(t) for t in f["huffman_code"]] for f in batch]
        max_len = max([len(code) for code in huffman_codes])
        mask = [[1] * len(code) + [0] * (max_len - len(code)) for code in huffman_codes]
        huffman_codes = [code + [0] * (max_len - len(code)) for code in huffman_codes]
        inputs_vector = torch.stack(inputs_vector, dim=0)
        path_nodes_indices = [
            f["path_nodes_index"] + [-1] * (max_len - len(f["huffman_code"]))
            for f in batch
        ]
        mask = torch.tensor(mask)
        return inputs_vector, path_nodes_indices, huffman_codes, mask


class CBOWSoftmaxDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        word2id_file: str,
        window_size: int = 5,
    ):
        super().__init__()
        self.window_size = window_size
        with open(file_path, "r") as f:
            self.sentences = json.load(f)
        self.word2id = json.load(open(word2id_file))
        self.id2word = {val: key for key, val in self.word2id.items()}
        self.data = self.data_process(sentences=self.sentences)
        self.words_num = len(self.word2id)

    def data_process(self, sentences):
        def remove_punctuation(text):
            text = re.sub(r"[^\w\s]", "", text)  # 去除标点符号
            return text

        sentences = [s for s in sentences if len(s) >= self.window_size]
        data = []
        for sentence in sentences:
            for i in range(len(sentence) - self.window_size + 1):
                t = sentence[i : i + self.window_size]
                t = [self.word2id[s] for s in t]
                data.append(t)
        return data

    def __getitem__(self, index):
        item = self.data[index].copy()
        target_id = item.pop(self.window_size // 2)  # item 将中心点去掉了
        input_vector = torch.zeros(self.words_num)
        input_vector[item] = 1
        return {"input_vector": input_vector, "target_id": target_id}

    def __len__(self):
        return len(self.data)

    def collate_fn(batch):
        inputs_vector = [f["input_vector"] for f in batch]
        target_ids = [f["target_id"] for f in batch]
        inputs_vector = torch.stack(inputs_vector, dim=0)
        target_ids = torch.LongTensor(target_ids)
        return inputs_vector, target_ids


class CBOWNegativeSamplingDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        word2id_file: str,
        word_freq_file: str,
        window_size: int = 5,
        neg_sample_size: int = 5,
        p_index: float = 0.75,
    ):
        super().__init__()
        self.window_size = window_size
        with open(file_path, "r") as f:
            self.sentences = json.load(f)
        self.word_freq = json.load(open(word_freq_file))
        self.word2id = json.load(open(word2id_file))
        self.id2word = {val: key for key, val in self.word2id.items()}
        self.words_num = len(self.word2id)
        self.neg_sample_size = neg_sample_size
        word_freq = [self.word_freq[self.id2word[i]] for i in range(self.words_num)]
        self.word_freq = torch.tensor(word_freq)
        self.word_weights = self.word_freq**p_index / (self.word_freq**p_index).sum()
        self.sampling_table = self._build_sampling_table(table_size=1e8)
        self.data, self.neg_samples = self.data_process(sentences=self.sentences)

    def _build_sampling_table(self, table_size=1e8):
        sampling_table = []
        for index, prob in enumerate(self.word_weights.tolist()):
            count = int(prob * table_size)
            sampling_table.extend([index] * count)
        return np.array(sampling_table)

    def neg_sampling(self, target_id):
        neg_sample_ids = np.random.choice(self.sampling_table, self.neg_sample_size)
        while target_id in neg_sample_ids:
            neg_sample_ids = np.random.choice(self.sampling_table, self.neg_sample_size)
        return neg_sample_ids

    def data_process(self, sentences):
        sentences = [s for s in sentences if len(s) >= self.window_size]
        data = []
        for sentence in sentences:
            for i in range(len(sentence) - self.window_size + 1):
                t = sentence[i : i + self.window_size]
                t = [self.word2id[s] for s in t]
                data.append(t)
        neg_samples = []
        for sample in tqdm(data, desc="Negative Sampling"):
            neg_sample = self.neg_sampling(sample[self.window_size // 2])
            neg_samples.append(neg_sample)
        return data, neg_samples

    def __getitem__(self, index):
        item = self.data[index].copy()
        target_id = item.pop(self.window_size // 2)  # item 将中心点去掉了
        # neg_sample_ids = self.neg_samples[index]
        neg_sample_ids = self.neg_sampling(target_id)
        input_vector = torch.zeros(self.words_num)
        input_vector[item] = 1
        chosen_ids = [target_id] + neg_sample_ids.tolist()
        label = [1] + [0] * self.neg_sample_size
        return {"input_vector": input_vector, "chosen_ids": chosen_ids, "label": label}

    def __len__(self):
        return len(self.data)

    def collate_fn(batch):
        inputs_vector = [f["input_vector"] for f in batch]
        chosen_ids = [f["chosen_ids"] for f in batch]
        labels = [f["label"] for f in batch]
        labels = torch.tensor(labels)
        inputs_vector = torch.stack(inputs_vector, dim=0)
        chosen_ids = torch.LongTensor(chosen_ids)
        return inputs_vector, chosen_ids, labels





if __name__ == "__main__":
    words_freq = json.load(open("Word2Vector/data/words_freq.json", "r"))
    huffman_root = construct_Huffman_tree(words_freq)
    huffman_codes = dict()
    no_leaf_codes = set()
    huffman_codes, no_leaf_codes = get_Huffman_codes(
        huffman_root, "", huffman_codes, no_leaf_codes
    )
    no_leaf_codes = {code: i for i, code in enumerate(no_leaf_codes)}
    with open("Word2Vector/data/huffman_codes.json", "w") as f:
        json.dump(
            huffman_codes,
            f,
            ensure_ascii=False,
        )
    with open("Word2Vector/data/no_leaf_code2index.json", "w") as f:
        json.dump(no_leaf_codes, f, ensure_ascii=False)
    print(len(words_freq))
    print(len(huffman_codes))
    if len(huffman_codes) + len(no_leaf_codes) == 2 * len(huffman_codes) - 1:
        print(True)
    else:
        print(False)
