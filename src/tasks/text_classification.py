from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from transformers import Trainer
from transformers import (
    HfArgumentParser,
    set_seed,
)
import os
import logging
from pathlib import Path
from src.common import read_jd_sentiment_cls_data
from src.common import TextClassificationDataSet
from .trainer import LocalTrainer 
from .data import EmbedCollator


from .arguments import ModelArguments, DataArguments, LocalTrainingArguments as TrainingArguments

logger = logging.getLogger(__name__)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = (labels == preds).sum().item() / len(labels)
    return {"accuracy": accuracy}
    

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    
    if (
        os.path.exists(training_args.output_dir) 
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir        
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # Set seed
    set_seed(training_args.seed)
    
    num_labels = 2
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False
    )
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=data_args.num_labels,
        cache_dir=model_args.cache_dir,
    )
    logger.info('Config: %s', config)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path, config=config
    )
    
    train_data = read_jd_sentiment_cls_data(data_args.train_data)
    eval_data = read_jd_sentiment_cls_data(data_args.eval_data)
    
    train_dataset = TextClassificationDataSet(train_data, tokenizer)
    eval_dataset = TextClassificationDataSet(eval_data, tokenizer)
    
    trainer = LocalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=EmbedCollator(
            tokenizer=tokenizer,            
        ),
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )
    
    print(model)
    trainer.train()
    
    results = trainer.evaluate()
    print(results)
    
    


if __name__ == "__main__":
    main()

"""
python -m src.tasks.text_classification \
--model_name_or_path /Users/mac/Downloads/robert-wwm \
--train_data /Users/mac/Desktop/learning_material/nlp_notes/data/text_classification/jd_sentiment_cls/train.csv \
--eval_data /Users/mac/Desktop/learning_material/nlp_notes/data/text_classification/jd_sentiment_cls/dev.csv \
--output_dir output 
--num_labels 1
"""


