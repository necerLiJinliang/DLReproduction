from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
import json
import pickle
import random
import torch
from tqdm import tqdm


class TransRDataset(Dataset):
    def __init__(
        self,
        data_dir,
        entity2id_file,
        relation2id_file,
        triples_file,
        mode,
        filtered=False,
        bern=False,
        advance=False,
        num_neg_samples=10,
        hpt_file="8_TransR/dataset/FB15k/hpt_r.pkl",
        tph_file="8_TransR/dataset/FB15k/tph_r.pkl",
    ):
        super(TransRDataset, self).__init__()
        self.data_dir = data_dir
        self.entity2id = self.load_json(entity2id_file)
        self.relation2id = self.load_json(relation2id_file)
        all_triples = self.load_pickle(triples_file)
        self.all_triples = set(self.triple2id(all_triples))
        self.data = pd.read_csv(data_dir, names=["h", "l", "t"], delimiter="\t")
        data_triples = self.get_triples()
        self.data_triples = self.triple2id(data_triples)
        self.filtered = filtered
        self.bern = bern
        self.num_neg_samples = num_neg_samples
        with open(hpt_file, "rb") as f:
            self.hpt_r = pickle.load(f)
        with open(tph_file, "rb") as f:
            self.tph_r = pickle.load(f)
        self.advance = advance
        self.get_h_t_num()

    def get_h_t_num(self):
        entity_h_num_dict = {val: 0 for _, val in self.entity2id.items()}
        entity_t_num_dict = {val: 0 for _, val in self.entity2id.items()}
        h_num = dict(self.data["h"].value_counts())
        t_num = dict(self.data["t"].value_counts())
        for key, val in h_num.items():
            entity_h_num_dict[self.entity2id[key]] = val
        for key, val in t_num.items():
            entity_t_num_dict[self.entity2id[key]] = val
        self.entity_h_num_dict = entity_h_num_dict
        self.entity_t_num_dict = entity_t_num_dict

    def load_json(self, path):
        with open(path, "r") as f:
            data = json.load(f)
        return data

    def load_pickle(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data

    def triple2id(self, triples):
        triples = [
            (self.entity2id[t[0]], self.relation2id[t[1]], self.entity2id[t[2]])
            for t in triples
        ]
        return triples

    def get_triples(self):
        triples = self.data.values.tolist()
        return triples

    def get_one_corrupted_data(self, h, r, t):
        entities = list(self.entity2id.values())
        cnt1 = 0
        cnt2 = 0
        h_num = self.entity_h_num_dict[h]
        t_num = self.entity_t_num_dict[t]

        if not self.bern:
            replace_h_prob = 0.5
        else:
            replace_h_prob = self.tph_r[r][1] / (self.tph_r[r][1] + self.hpt_r[r][1])

        neg_samples = []
        # 谁的频率小，倾向于替换谁
        while len(neg_samples) < self.num_neg_samples:
            sample = []
            p = np.random.random(1)[0]
            if p < replace_h_prob:
                # 替换头实体
                # corrupted_h = random.sample(entities, 1)[0]
                corrupted_h = random.choice(entities)
                while corrupted_h == h:
                    # corrupted_h = random.sample(entities, 1)[0]
                    corrupted_h = random.choice(entities)
                sample.append(corrupted_h)
                sample.append(t)
            else:
                # 替换尾实体
                # corrupted_t = random.sample(entities, 1)[0]
                corrupted_t = random.choice(entities)
                while corrupted_t == t:
                    # corrupted_t = random.sample(entities, 1)[0]
                    corrupted_t = random.choice(entities)
                sample.append(h)
                sample.append(corrupted_t)
            neg_samples.append(sample)
        return neg_samples

    def __len__(self):
        return len(self.data_triples)

    def __getitem__(self, idx):
        h, r, t = self.data_triples[idx]
        neg_samples = self.get_one_corrupted_data(h, r, t)
        return {
            "h": h,
            "r": r,
            "t": t,
            "neg_samples": neg_samples,
        }

    def collate_fn(batch):
        h = [f["h"] for f in batch]
        r = [f["r"] for f in batch]
        t = [f["t"] for f in batch]
        neg_samples = [f["neg_samples"] for f in batch]
        return {
            "h": torch.tensor(h, dtype=torch.long),
            "r": torch.tensor(r, dtype=torch.long),
            "t": torch.tensor(t, dtype=torch.long),
            "neg_samples": torch.tensor(neg_samples, dtype=torch.long),
        }


def get_meta_data(
    data_folder_path="dataset/FB15k",
    train_data_path="train.txt",
    valid_dat_path="valid.txt",
    test_data_path="test.txt",
    entity2id_path="entity2id.json",
    label2id_path="label2id.json",
    all_triples_path="all_triples.pkl",
):
    print(f"Get meta data from {data_folder_path}")
    train_data_path = os.path.join(data_folder_path, train_data_path)
    valid_dat_path = os.path.join(data_folder_path, valid_dat_path)
    test_data_path = os.path.join(data_folder_path, test_data_path)
    entity2id_path = os.path.join(data_folder_path, entity2id_path)
    label2id_path = os.path.join(data_folder_path, label2id_path)
    all_triples_path = os.path.join(data_folder_path, all_triples_path)
    train_data = pd.read_csv(train_data_path, delimiter="\t", names=["h", "r", "t"])
    valid_data = pd.read_csv(valid_dat_path, delimiter="\t", names=["h", "r", "t"])
    test_data = pd.read_csv(test_data_path, delimiter="\t", names=["h", "r", "t"])
    all_data = pd.concat([train_data, test_data, valid_data], axis=0)
    all_entities = set(
        all_data["h"].unique().tolist() + all_data["t"].unique().tolist()
    )
    entity2id = {entity: i for i, entity in enumerate(all_entities)}
    # 获取所有的关系，label2id
    all_labels = set(all_data["r"].tolist())
    label2id = {label: i for i, label in enumerate(all_labels)}
    with open(entity2id_path, "w") as f:
        json.dump(entity2id, f, ensure_ascii=False)
    with open(label2id_path, "w") as f:
        json.dump(label2id, f, ensure_ascii=False)
    all_triples = all_data.drop_duplicates().values.tolist()
    all_triples = [tuple(s) for s in all_triples]
    all_triples = set(all_triples)
    with open(all_triples_path, "wb") as f:
        pickle.dump(all_triples, f)

    rel2id = json.load(open(os.path.join(data_folder_path, "label2id.json")))
    data_r = all_data["r"].unique().tolist()
    hpt_r = dict()
    tph_r = dict()
    for r in tqdm(data_r):
        hpt = all_data["h"].loc[all_data["r"] == r].value_counts()
        hpt = hpt.sum() / hpt.shape[0]
        hpt_r[rel2id[r]] = hpt
        tph = all_data["t"].loc[all_data["r"] == r].value_counts()
        tph = tph.sum() / tph.shape[0]
        tph_r[rel2id[r]] = tph
    hpt_r_list = sorted(hpt_r.items(), key=lambda x: x[0])
    tph_r_list = sorted(tph_r.items(), key=lambda x: x[0])
    with open(os.path.join(data_folder_path, "hpt_r.pkl"), "wb") as f:
        pickle.dump(hpt_r_list, f)
    with open(os.path.join(data_folder_path, "tph_r.pkl"), "wb") as f:
        pickle.dump(tph_r_list, f)


if __name__ == "__main__":
    get_meta_data(data_folder_path="8_TransR/dataset/FB15k/")
