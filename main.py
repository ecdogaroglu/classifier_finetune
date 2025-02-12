# Description: Main script to run the training pipeline

from config import ModelConfig
from classifier import TextClassifier, logger

def main():
    try:
        config = ModelConfig()
        classifier = TextClassifier(config)
        
        # Load and prepare data
        df = classifier.load_data("data.xlsx")
        tokenized_datasets, class_weights = classifier.prepare_data(df)
        
        # Train model
        eval_results, test_results = classifier.train(tokenized_datasets, class_weights)
        
        logger.info("Training pipeline completed successfully")
        logger.info(f"Final evaluation results: {eval_results}")
        logger.info(f"Final test results: {test_results}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()