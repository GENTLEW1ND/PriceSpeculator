from mlProject.config.configuration import *
from mlProject.components.data_transformation import *
from mlProject import logger

Stage_Name = "Data Transformation stage"

class DataTransformaitonTrainingPipeline:
    def __init__(self):
        pass
    def main(self): 
         config = ConfigurationManager()
         data_transformation_config = config.get_data_transformation_config()
         data_transformation = DataTransformation(data_transformation_config)
         transformed_data = data_transformation.Transformation()
         data_transformation.TrainTestSplit(transformed_data)
            
if __name__ == '__main__':
    try:
        logger.info(f" stage {Stage_Name} started")
        obj = DataTransformaitonTrainingPipeline()
        obj.main()
        logger.info(f"stage {Stage_Name} completed")
    except Exception as e:
        logger.exception(e)
        raise e