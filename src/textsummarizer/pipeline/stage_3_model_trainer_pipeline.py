from src.textsummarizer.config.configuration import ConfigurationManager
from src.textsummarizer.components.model_trainer import ModelTrainer
from src.textsummarizer.logging import logging

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def initiate_model_training(self):
        try:
            config = ConfigurationManager()
            model_trainer_config = config.get_model_trainer_config()
            model_trainer = ModelTrainer(config=model_trainer_config)
            # model_trainer.train()
        except Exception as e:
            raise e