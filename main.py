from mlProject import logger
from mlProject.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
                            
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