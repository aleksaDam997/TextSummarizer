import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from src.textsummarizer.entity import ModelTrainerConfig

class ModelTrainer: 
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def train(self):
        
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        is_main_process = local_rank in [-1, 0]

        if is_main_process:
            print(f"GPUs: {torch.cuda.device_count()}, World size: {world_size}, Rank: {local_rank}")

        model_name = self.config.model_ckpt
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # FIX: fp32 za stabilnost — Pegasus ima overflow probleme u fp16
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
        model.config.use_cache = False

        lora_config = LoraConfig(
            r = self.config.lora_parameters.r,
            lora_alpha = self.config.lora_parameters.lora_alpha,
            target_modules = self.config.lora_parameters.target_modules,
            lora_dropout = self.config.lora_parameters.lora_dropout,
            bias= self.config.lora_parameters.bias,
            task_type = self.config.lora_parameters.task_type,
        )
        model = get_peft_model(model, lora_config)

        if is_main_process:
            model.print_trainable_parameters()

        dataset_samsum = load_from_disk(self.config.data_path)

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model,
            padding=True, label_pad_token_id=-100,
        )

        training_args = Seq2SeqTrainingArguments(
            output_dir="/kaggle/working/pegasus-lora",
            num_train_epochs=1,
            per_device_train_batch_size=2,      # manji batch za stabilnost
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=3e-5,                 # FIX: manji LR (1e-4 → 3e-5)
            warmup_ratio=0.05,                  # FIX: ratio umjesto steps, sigurniji
            weight_decay=0.01,
            max_grad_norm=1.0,
            fp16=False,                         # FIX: isključen fp16
            bf16=False,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=200,
            save_steps=500,
            save_total_limit=2,
            predict_with_generate=True,
            generation_max_length=128,
            report_to="none",
            remove_unused_columns=False,
            label_names=["labels"],
            ddp_find_unused_parameters=False,
            dataloader_num_workers=2,
        )

        dataset_samsum_pt = load_from_disk(self.config.data_path)

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            processing_class=tokenizer,
            data_collator=data_collator,
            train_dataset=dataset_samsum_pt["train"],
            eval_dataset=dataset_samsum_pt["validation"],
        )

        trainer.train()

        # trainer.save_model("/kaggle/working/pegasus-lora-adapter")
        # tokenizer.save_pretrained("/kaggle/working/pegasus-lora-adapter")
        # print("Saved!")

        merged_model = model.merge_and_unload()
        print(type(merged_model))  # sad je čisti PegasusForConditionalGeneration

        # Snimi merged model
        merged_model.save_pretrained(os.path.join(self.config.output_dir, self.config.pegasus-lora-merged))
