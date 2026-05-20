from src.textsummarizer.config.configuration import ConfigurationManager
from src.textsummarizer.components.model_evaluation import ModelEvaluation

class ModelEvaluationPipeline:

    def __init__(self):
        pass

    def initiate_model_evaluation(self):
        try:
            config = ConfigurationManager()
            model_evaluation_config = config.get_model_evaluation_config()
            model_evaluation = ModelEvaluation(config=model_evaluation_config)
            model_evaluation.evaluate()
        except Exception as e:
            raise e