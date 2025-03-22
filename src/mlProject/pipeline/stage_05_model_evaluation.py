from mlProject.config.configuration import *
from mlProject.components.model_evaluation import *
from mlProject import logger

Stage_Name = "Model Evaluation stage"

class ModelEvaluationPipeline:
    def __init__(self):
        pass
    def main(self): 
         config = ConfigurationManager()
         model_evaluation_config = config.get_model_evaluation_config()
         model_evaluation = ModelEvaluation(config=model_evaluation_config)
         model_evaluation.save_results()
            
if __name__ == '__main__':
    try:
        logger.info(f" stage {Stage_Name} started")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f"stage {Stage_Name} completed")
    except Exception as e:
        logger.exception(e)
        raise e