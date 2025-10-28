import jieba
import re
import json
from collections import Counter
from my_utils import construct_Huffman_tree, get_Huffman_codes
import argparse


def remove_punctuation(text):
    text = re.sub(r"[^\w\s]", "", text)  # 去除标点符号
    return text


def process_txt(
    text_file: str,
    processed_file: str,
    words_freq_file: str,
    word2id_file: str,
):
    """文本预处理函数

    Args:
        text_file (str): 原txt文本路径
        processed_file (str): 处理后的文本存储路径
        words_freq_file (str): 词频表存储路径
        word2id_file (str): 词语id表存储路径
    """
    with open(text_file) as f:
        text = f.read()
    sentences = text.split("。")
    sentences = [remove_punctuation(sentence) for sentence in sentences]
    sentences = [jieba.lcut(s) for s in sentences]
    sentences = [
        [w for w in words if w != "\n" and w != " " and w != "\u3000"]
        for words in sentences
    ]
    words = sum(sentences, [])
    words_freq = Counter(words)
    word2id = {key: i for i, key in enumerate(words_freq)}
    words_freq_dict = dict(words_freq)
    with open(processed_file, "w") as f:
        json.dump(sentences, f, ensure_ascii=False)
    with open(
        words_freq_file,
        "w",
    ) as f:
        json.dump(words_freq_dict, f, ensure_ascii=False)
    with open(word2id_file, "w") as f:
        json.dump(word2id, f, ensure_ascii=False)


def format_huffman_data(
    words_freq_file: str,
    huffman_codes_file: str,
    no_leaf_code2index_file: str,
):
    """构造构造huffman树，获取huffman编码，非叶子节点编码并存储相关数据

    Args:
        words_freq_file (str): 词频表
        huffman_codes_file (str): huffman编码存储路径
        no_leaf_code2index_file (str): 非叶子节点编码存储路径
    """
    words_freq = json.load(open(words_freq_file, "r"))
    huffman_root = construct_Huffman_tree(words_freq)
    huffman_codes = dict()
    no_leaf_codes = set()
    huffman_codes, no_leaf_codes = get_Huffman_codes(
        huffman_root, "", huffman_codes, no_leaf_codes
    )
    no_leaf_codes = {code: i for i, code in enumerate(no_leaf_codes)}
    with open(huffman_codes_file, "w") as f:
        json.dump(
            huffman_codes,
            f,
            ensure_ascii=False,
        )
    with open(no_leaf_code2index_file, "w") as f:
        json.dump(no_leaf_codes, f, ensure_ascii=False)
    assert len(huffman_codes) + len(no_leaf_codes) == 2 * len(huffman_codes) - 1
    # print(len(words_freq))
    # print(len(huffman_codes))
    # if len(huffman_codes) + len(no_leaf_codes) == 2 * len(huffman_codes) - 1:
    #     print(True)
    # else:
    #     print(False)


def main():
    parser = argparse.ArgumentParser(description="Process text data.")
    parser.add_argument(
        "--text_file",
        type=str,
        default="Word2Vector/data/data.txt",
        help="Origin text data file path.",
    )
    parser.add_argument(
        "--words_freq_file",
        type=str,
        default="Word2Vector/data/words_freq.json",
        help="File path of frequency table of words.",
    )
    parser.add_argument(
        "--processed_file",
        type=str,
        default="Word2Vector/data/processed_data.json",
        help="Data file of processed data.",
    )
    parser.add_argument(
        "--word2id_file",
        type=str,
        default="Word2Vector/data/word2id.json",
        help="Word to id table file path.",
    )
    parser.add_argument(
        "--huffman_codes_file",
        type=str,
        default="Word2Vector/data/huffman_codes.json",
        help="Huffman codes file path.",
    )
    parser.add_argument(
        "--no_leaf_code2index_file",
        type=str,
        default="Word2Vector/data/no_leaf_code2index.json",
        help="No leaf node to index file path.",
    )
    args = parser.parse_args()

    process_txt(
        text_file=args.text_file,
        processed_file=args.processed_file,
        words_freq_file=args.words_freq_file,
        word2id_file=args.word2id_file,
    )
    format_huffman_data(
        words_freq_file=args.words_freq_file,
        huffman_codes_file=args.huffman_codes_file,
        no_leaf_code2index_file=args.no_leaf_code2index_file,
    )
    print(True)


if __name__ == "__main__":
    main()
