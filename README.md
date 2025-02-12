# Text Classification with DistilRoBERTa

This repository contains a robust implementation of a text classification system using DistilRoBERTa, designed to handle imbalanced datasets with multiple classes. The implementation includes features such as proper data preprocessing, class balancing, model training with early stopping, and comprehensive metric logging using MLflow.

## Features

- **Robust Data Handling**
  - Automatic handling of imbalanced classes
  - Stratified data splitting
  - Configurable resampling strategies
  - Support for large datasets

- **Advanced Model Training**
  - Selective layer freezing for transfer learning
  - Early stopping to prevent overfitting
  - Gradient accumulation for effective batch processing
  - Class-weighted training for imbalanced datasets

- **Comprehensive Monitoring**
  - Detailed per-class performance metrics
  - MLflow integration for experiment tracking
  - Proper error handling and logging
  - Progress tracking during training

## Prerequisites

```bash
pip install torch transformers datasets pandas scikit-learn mlflow
```

## Project Structure

```
.
├── README.md           # This file
├── config.py           # Model configurations
├── main.py             # Main implementation file
├── classifier.py       # Class representing data processing, model creation, training and metric tracking
└── mlruns/             # MLflow tracking directory (once the code is run)
```

## Usage

1. **Data Preparation**
   
   Prepare your data in Excel format with at least two columns:
   - `text`: The input text to be classified
   - `category`: The category/label for the text

2. **Configuration**

   Modify the `ModelConfig` class parameters in the code:

   ```python
   @dataclass
   class ModelConfig:
       model_name: str = "valurank/distilroberta-topic-classification"
       max_length: int = 128
       batch_size: int = 16
       num_epochs: int = 5
       learning_rate: float = 3e-4
       weight_decay: float = 0.01
       warmup_ratio: float = 0.1
       min_samples_per_class: int = 100
       max_samples_per_class: int = 1000
       frozen_layers: int = 10
   ```

3. **Running the Training**

   ```python
   from distillroberta import TextClassifier, ModelConfig

   # Initialize and train
   config = ModelConfig()
   classifier = TextClassifier(config)
   
   # Load and prepare data
   df = classifier.load_data("your_data.xlsx")
   tokenized_datasets, class_weights = classifier.prepare_data(df)
   
   # Train model
   eval_results, test_results = classifier.train(tokenized_datasets, class_weights)
   ```

## Key Components

### TextClassifier Class

The main class that handles the entire pipeline:

- Data loading and preprocessing
- Model initialization and configuration
- Training and evaluation
- Metric logging

### Data Processing

- Handles class imbalance through resampling
- Implements proper stratification for data splitting
- Provides robust error handling for data loading

### Model Training

- Implements early stopping
- Uses gradient accumulation for stable training
- Provides comprehensive progress tracking
- Handles model checkpointing

### Metric Logging

- Tracks accuracy, precision, and F1 scores
- Provides per-class performance metrics
- Integrates with MLflow for experiment tracking

## Best Practices

1. **Data Preparation**
   - Clean your data before training
   - Remove duplicates and handle missing values
   - Ensure consistent labeling

2. **Model Configuration**
   - Adjust batch size based on available memory
   - Tune learning rate and weight decay if needed
   - Modify frozen layers based on your dataset size

3. **Training**
   - Monitor training progress using MLflow
   - Check per-class metrics for imbalanced datasets
   - Use early stopping to prevent overfitting

## Troubleshooting

Common issues and solutions:

1. **Stagnant Loss / Unstable Gradients**
   - Adjust the learning rate
   - Use gradient clipping with max_grad_norm
   - Unfreeze more encoder layers

2. **Memory Issues**
   - Reduce batch size
   - Decrease maximum sequence length
   - Use gradient accumulation

3. **Class Imbalance**
   - Adjust min_samples_per_class and max_samples_per_class
   - Monitor per-class metrics
   - Consider modifying class weights

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.
