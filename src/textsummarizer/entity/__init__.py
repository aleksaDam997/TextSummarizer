from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from typing import List
from peft import TaskType
from typing import Literal
@dataclass
class DataIngestionConfig: 
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    tokenizer_name: str



@dataclass
class TrainingParams:
    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_ratio: float
    weight_decay: float
    max_grad_norm: float

    fp16: bool
    bf16: bool

    logging_steps: int
    evaluation_strategy: Literal["no", "steps", "epoch"]
    eval_steps: int
    save_steps: int
    save_total_limit: int

    predict_with_generate: bool
    generation_max_length: int

    report_to: str
    remove_unused_columns: bool
    label_names: tuple

    ddp_find_unused_parameters: bool
    dataloader_num_workers: int
    local_rank: int = -1

@dataclass
class LoraParameters:
    r: int
    lora_alpha: int
    target_modules: List[str]
    lora_dropout: float
    bias: str
    task_type: TaskType

@dataclass
class ModelTrainerConfig:
    output_dir: Path
    model_ckpt: Path
    data_path: Path
    training_params: TrainingParams
    lora_parameters: LoraParameters


@dataclass(frozen = True)
class ModelEvaluationConfig:
    root_dir: Path
    data_path: Path
    model_path: Path
    tokenizer_path: Path
    metric_file_name: Path

    