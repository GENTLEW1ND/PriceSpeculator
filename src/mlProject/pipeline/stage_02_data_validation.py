from mlProject.config.configuration import *
from mlProject.components.data_validation import *
from mlProject import logger

Stage_Name = "Data validation stage"

class DataValidationTrainingPipeline:
    def __init__(self):
        pass
    def main(self): 
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        data_validation.validate_all_columns()
            
if __name__ == '__main__':
    try:
        logger.info(f" stage {Stage_Name} started")
        obj = DataValidationTrainingPipeline()
        obj.main()
        logger.info(f"stage {Stage_Name} completed")
    except Exception as e:
        logger.exception(e)
        raise e