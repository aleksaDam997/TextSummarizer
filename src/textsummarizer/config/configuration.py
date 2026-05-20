from src.textsummarizer.constants import *
from src.textsummarizer.utils.common import read_yaml, create_directories
from src.textsummarizer.entity import DataIngestionConfig, DataTransformationConfig, ModelTrainerConfig, ModelEvaluationConfig, TrainingParams, LoraParameters


class ConfigurationManager:
    def __init__(self, config_path=CONFIG_FILE_PATH, params_file_path=PARAMS_FILE_PATH):
        self.config = read_yaml(config_path)
        self.params = read_yaml(params_file_path)

        create_directories([self.config.artifact_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir)
        )

        return data_ingestion_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config=self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(root_dir=config.root_dir,
            data_path = config.data_path, tokenizer_name = config.tokenizer_name)

        return data_transformation_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        trainingParams = self.params.TrainingArguments
        loraParams = self.params.Lora

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
                output_dir = config.root_dir,
                model_ckpt=config.model_ckpt,
                data_path= config.data_path,

                training_params = TrainingParams(
                    num_train_epochs = trainingParams.num_train_epochs,
                    per_device_train_batch_size = trainingParams.per_device_train_batch_size,
                    per_device_eval_batch_size = trainingParams.per_device_eval_batch_size,
                    gradient_accumulation_steps = trainingParams.gradient_accumulation_steps,
                    learning_rate = trainingParams.learning_rate,
                    warmup_ratio = trainingParams.warmup_ratio,
                    weight_decay = trainingParams.weight_decay,
                    max_grad_norm = trainingParams.max_grad_norm,
                    fp16 = trainingParams.fp16,
                    bf16 = trainingParams.bf16,
                    logging_steps = trainingParams.logging_steps,
                    evaluation_strategy = trainingParams.eval_strategy,
                    eval_steps = trainingParams.eval_steps,
                    save_steps = trainingParams.save_steps,
                    save_total_limit = trainingParams.save_total_limit,
                    predict_with_generate = trainingParams.predict_with_generate,
                    generation_max_length = trainingParams.generation_max_length,
                    report_to = trainingParams.report_to,
                    remove_unused_columns = trainingParams.remove_unused_columns,
                    label_names = trainingParams.label_names,
                    ddp_find_unused_parameters = trainingParams.ddp_find_unused_parameters,
                    dataloader_num_workers = trainingParams.dataloader_num_workers
            ),
            lora_parameters = LoraParameters(
                r = loraParams.r,
                lora_alpha = loraParams.lora_alpha,
                target_modules = loraParams.target_modules,
                lora_dropout = loraParams.lora_dropout,
                bias = loraParams.bias,
                task_type = loraParams.task_type
            )
        )

        return model_trainer_config
    
    def get_model_evaluation_config(self)-> ModelEvaluationConfig:
    
        config = self.config.model_evaluation

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir = config.root_dir,
            data_path = config.data_path,
            model_path = config.model_path,
            tokenizer_path = config.tokenizer_path,
            metric_file_name = config.metric_file_name
        )

        return model_evaluation_config