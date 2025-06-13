import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

pad_token_id = 0


class GSM8KDateset(torch.utils.data.Dataset):
    """Some Information about GSM8KDateset"""

    def __init__(self, data, tokenizer):
        super(GSM8KDateset, self).__init__()
        self.tokenizer = tokenizer
        self.question, self.answer, self.thought_answer, self.length = (
            self._pre_process(data)
        )

    def _pre_process(self, data):
        question = data["question"]
        thought_answer = data["answer"]
        answer = ["### " + t.split("#### ")[-1] for t in thought_answer]
        length = len(question)
        question = self.tokenizer(
            question, max_length=512, truncation=True, add_special_tokens=True
        )
        answer = self.tokenizer(
            answer, max_length=512, truncation=True, add_special_tokens=True
        )
        thought_answer = self.tokenizer(
            thought_answer, max_length=512, truncation=True, add_special_tokens=True
        )
        return question, answer, thought_answer, length

    def __getitem__(self, index):
        question_ids = self.question["input_ids"][index]
        question_attention_mask = self.question["attention_mask"][index]
        answer_ids = self.answer["input_ids"][index]
        answer_attention_mask = self.answer["attention_mask"][index]
        thought_answer_ids = self.thought_answer["input_ids"][index]
        thought_answer_attention_mask = self.thought_answer["attention_mask"][index]
        return {
            "question_ids": question_ids,
            "question_attention_mask": question_attention_mask,
            "answer_ids": answer_ids,
            "answer_attention_mask": answer_attention_mask,
            "thought_answer_ids": thought_answer_ids,
            "thought_answer_attention_mask": thought_answer_attention_mask,
        }

    def __len__(self):
        return self.length

    def collate_fn(batch):
        question_ids = [f["question_ids"] for f in batch]
        question_attention_mask = [f["question_attention_mask"] for f in batch]
        question_max_len = max([len(f) for f in question_ids])
        question_ids = [
            f + [pad_token_id] * (question_max_len - len(f)) for f in question_ids
        ]
        question_attention_mask = [
            f + [0] * (question_max_len - len(f)) for f in question_attention_mask
        ]
        answer_ids = [f["answer_ids"] for f in batch]
        answer_max_len = max([len(f) for f in answer_ids])
        answer_ids = [
            f + [pad_token_id] * (answer_max_len - len(f)) for f in answer_ids
        ]
        thought_answer_ids = [f["thought_answer_ids"] for f in batch]
        thought_answer_max_len = max([len(f) for f in thought_answer_ids])
        thought_answer_ids = [
            f + [pad_token_id] * (thought_answer_max_len - len(f))
            for f in thought_answer_ids
        ]
        return {
            "question_ids": torch.tensor(question_ids),
            "question_attention_mask": torch.tensor(question_attention_mask),
            "answer_ids": torch.tensor(answer_ids),
            "thought_answer_ids": torch.tensor(thought_answer_ids),
        }
