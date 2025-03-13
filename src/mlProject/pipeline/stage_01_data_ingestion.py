from mlProject.config.configuration import *
from mlProject.components.data_ingestion import *
from mlProject import logger

Stage_Name = "Data ingestion stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass
    def main(self): 
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()
        
if __name__ == '__main__':
    try:
        logger.info(f" stage {Stage_Name} started")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f"stage {Stage_Name} completed")
    except Exception as e:
        logger.exception(e)
        raise e