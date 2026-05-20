from src.textsummarizer.logging import logger
from src.textsummarizer.pipeline.stage_1_data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.textsummarizer.pipeline.stage_2_data_transformation_pipeline import DataTransformationTrainingPipeline
from src.textsummarizer.pipeline.stage_3_model_trainer_pipeline import ModelTrainingPipeline
from src.textsummarizer.pipeline.stage_4_model_evaluation_pipeline import ModelEvaluationPipeline

STAGE_NAME = "Data Ingestion stage"

try:
    logger.info(f'stage {STAGE_NAME} initiated')
    data_ingestion_pipeline = DataIngestionTrainingPipeline()
    data_ingestion_pipeline.initiate_data_ingestion()
    logger.info(f"Stage {STAGE_NAME} completed.")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Transformation stage"

try:
    logger.info(f'stage {STAGE_NAME} initiated')
    data_transformation_pipeline = DataTransformationTrainingPipeline()
    data_transformation_pipeline.initiate_data_transformation()
    logger.info(f"Stage {STAGE_NAME} completed.")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model training stage"

try:
    logger.info(f'stage {STAGE_NAME} initiated')
    model_training_pipeline = ModelTrainingPipeline()
    model_training_pipeline.initiate_model_training()
    logger.info(f"Stage {STAGE_NAME} completed.")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model evaluation stage"

try:
    logger.info(f'stage {STAGE_NAME} initiated')
    model_evaluation_pipeline = ModelEvaluationPipeline()
    model_evaluation_pipeline.initiate_model_evaluation()
    logger.info(f"Stage {STAGE_NAME} completed.")
except Exception as e:
    logger.exception(e)
    raise e