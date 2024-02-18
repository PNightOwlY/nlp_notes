import os
from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments

@dataclass
class ModelArguments:
    
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    
@dataclass
class DataArguments:
    train_data: str = field(
        default=None, metadata={"help": "Path to train data"}
    )
    
    eval_data: str = field(
        default=None, metadata={"help": "Path to eval data"}
    )
    
    num_labels: int = field(default=2)
    
    query_max_len: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    
    passage_max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    
    def __post_init__(self):
        if not os.path.exists(self.train_data):
            raise FileNotFoundError(f"cannot find train data file:{self.train_data}, please set a true path")
        
@dataclass
class LocalTrainingArguments(TrainingArguments):
    temperature: Optional[float] = field(default=0.02)
    normlized: bool = field(default=True)
    sentence_pooling_method: str = field(default='cls', metadata={"help": "the pooling method, should be cls or mean"})


