from inspect import signature

from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
from textSummarizer.entity import ModelTrainerConfig
import torch
import os


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config


    
    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Check if model already exists
        model_path = os.path.join(self.config.root_dir, "pegasus-samsum-model")
        tokenizer_path = os.path.join(self.config.root_dir, "tokenizer")
        
        if os.path.exists(model_path):
            print(f"Loading pre-trained model from {model_path}")
            # Try to load tokenizer from saved path, fall back to model_path or base model if not found
            if os.path.exists(tokenizer_path):
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
            else:
                print(f"Tokenizer not found at {tokenizer_path}, checking {model_path} or base model")
                if os.path.exists(os.path.join(model_path, "tokenizer_config.json")):
                    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
                tokenizer.save_pretrained(tokenizer_path)
            model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True).to(device)
            return  # Skip training if model already exists
        
        # Load base model and train
        print(f"Loading base model from {self.config.model_ckpt}")
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)
        
        #loading data 
        dataset_samsum_pt = load_from_disk(self.config.data_path)

        training_args_kwargs = dict(
            output_dir=self.config.root_dir,
            num_train_epochs=self.config.num_train_epochs,
            warmup_steps=self.config.warmup_steps,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
        )

        training_args_signature = signature(TrainingArguments.__init__)
        if "eval_strategy" in training_args_signature.parameters:
            training_args_kwargs["eval_strategy"] = self.config.evaluation_strategy
        elif "evaluation_strategy" in training_args_signature.parameters:
            training_args_kwargs["evaluation_strategy"] = self.config.evaluation_strategy

        trainer_args = TrainingArguments(**training_args_kwargs)

        trainer = Trainer(
            model=model_pegasus,
            args=trainer_args,
            data_collator=seq2seq_data_collator,
            train_dataset=dataset_samsum_pt["train"],
            eval_dataset=dataset_samsum_pt["validation"],
            processing_class=tokenizer,
        )
        
        trainer.train()

        ## Save model
        model_pegasus.save_pretrained(os.path.join(self.config.root_dir,"pegasus-samsum-model"))
        ## Save tokenizer
        tokenizer.save_pretrained(os.path.join(self.config.root_dir,"tokenizer"))
