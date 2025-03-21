from mlProject.config.configuration import *
from mlProject.components.model_trainer import *
from mlProject import logger

Stage_Name = "Model Training stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass
    def main(self): 
         config = ConfigurationManager()
         model_trainer_config = config.get_model_trainer_config()
         model_trainer = ModelTrainer(config = model_trainer_config)
         model_trainer.train()
            
if __name__ == '__main__':
    try:
        logger.info(f" stage {Stage_Name} started")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f"stage {Stage_Name} completed")
    except Exception as e:
        logger.exception(e)
        raise e