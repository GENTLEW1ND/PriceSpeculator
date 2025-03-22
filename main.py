from mlProject import logger
from mlProject.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from mlProject.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from mlProject.pipeline.stage_03_data_transformation import DataTransformaitonTrainingPipeline
from mlProject.pipeline.stage_04_model_trainer import ModelTrainingPipeline
from mlProject.pipeline.stage_05_model_evaluation import ModelEvaluationPipeline
                            
Stage_Name = "Data ingestion stage"
                        
if __name__ == '__main__':
    try:
        logger.info(f" stage {Stage_Name} started")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f"stage {Stage_Name} completed")
    except Exception as e:
        logger.exception(e)
        raise e
    
Stage_Name = "Data validation stage"

if __name__ == '__main__':
    try:
        logger.info(f"stage {Stage_Name} started")
        data_validation = DataValidationTrainingPipeline()
        data_validation.main()
        logger.info(f"Stage {Stage_Name} completed")
    except Exception as e:
        logger.exception(e)
        raise e

Stage_Name = "Data Transformation stage"

if __name__ == '__main__':
    try:
        logger.info(f" stage {Stage_Name} started")
        obj = DataTransformaitonTrainingPipeline()
        obj.main()
        logger.info(f"stage {Stage_Name} completed")
    except Exception as e:
        logger.exception(e)
        raise e
    
Stage_Name = "Model Training stage"

if __name__ == '__main__':
    try:
        logger.info(f" stage {Stage_Name} started")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f"stage {Stage_Name} completed")
    except Exception as e:
        logger.exception(e)
        raise e
    
Stage_Name = "Model Evaluation stage"

if __name__ == '__main__':
    try:
        logger.info(f" stage {Stage_Name} started")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f"stage {Stage_Name} completed")
    except Exception as e:
        logger.exception(e)
        raise e