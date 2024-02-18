from transformers.trainer import *

class LocalTrainer(Trainer):
    def _save(self, output_dir: Optional[str]=None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Saving model checkpoints to {output_dir}")
        
        if not hasattr(self.model , 'save'):
            raise NotImplementedError(
                f'MODEL {self.model.__class__.__name__} does not support save interface'
            )
            
        else:
            self.model.save(output_dir)
        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        print(inputs)
        outputs = model(**inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss