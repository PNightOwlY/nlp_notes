from dataclasses import dataclass
from typing import List, Tuple

from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizer
import torch




@dataclass
class EmbedCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    query_max_len: int = 32
    passage_max_len: int = 128

    def padding_score(self, teacher_score):
        group_size = None
        for scores in teacher_score:
            if scores is not None:
                group_size = len(scores)
                break
        if group_size is None:
            return None

        padding_scores = [100.0] + [0.0] * (group_size - 1)
        new_teacher_score = []
        for scores in teacher_score:
            if scores is None:
                new_teacher_score.append(padding_scores)
            else:
                new_teacher_score.append(scores)
        return new_teacher_score

    def __call__(self, features):
        
        # print(features)
        text = [f[0] for f in features]
        label = torch.tensor([f[1] for f in features], dtype=torch.long)

        if isinstance(text[0], list):
            text = sum(text, [])
        if isinstance(label[0], list):
            label = sum(label, [])
        print(text)
        q_collated = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors="pt",
        )
        # d_collated = self.tokenizer(
        #     passage,
        #     padding=True,
        #     truncation=True,
        #     max_length=self.passage_max_len,
        #     return_tensors="pt",
        # )
        return {
            "input_ids": q_collated['input_ids'],
            "attention_mask": q_collated['attention_mask'],
            "labels": label
        }