# Description: This file contains the implementation of the TextClassifier class, which is responsible for loading, preprocessing, and training a text classification model using the Hugging Face Transformers library.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
import mlflow
import torch
import numpy as np
from typing import Tuple, List
import logging

from config import ModelConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextClassifier:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.label_encoder = LabelEncoder()

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load and preprocess the data."""
        try:
            df = pd.read_excel(file_path)
            if not all(col in df.columns for col in ['text', 'category']):
                raise ValueError("Required columns 'text' and 'category' not found in the dataset")
            
            df = (df
                 .dropna(subset=['text', 'category'])
                 .drop_duplicates())
            
            logger.info(f"Loaded dataset with {len(df)} samples")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def prepare_data(self, df: pd.DataFrame) -> Tuple[DatasetDict, List[str]]:
        """Prepare data for training."""
        # Encode labels
        df['category'] = self.label_encoder.fit_transform(df['category'])
        class_weights = self._calculate_class_weights(df['category'])
        
        # Split data with proper stratification
        train_df, temp_df = self._stratified_split(df, test_size=0.3)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
        
        # Resample training data
        train_df = self._resample_data(train_df)
        
        # Create dataset dictionary
        datasets = DatasetDict({
            "train": Dataset.from_pandas(train_df),
            "validation": Dataset.from_pandas(val_df),
            "test": Dataset.from_pandas(test_df)
        })
        
        # Tokenize datasets
        tokenized_datasets = self._tokenize_datasets(datasets)
        
        return tokenized_datasets, class_weights

    def _calculate_class_weights(self, labels: pd.Series) -> List[float]:
        """Calculate class weights for balanced training."""
        class_counts = labels.value_counts()
        total_samples = len(labels)
        return [total_samples / (len(class_counts) * count) for count in class_counts]

    def _stratified_split(self, df: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Perform stratified split handling minority classes."""
        min_samples = 8
        class_counts = df['category'].value_counts()
        stratifiable_classes = class_counts[class_counts >= min_samples].index
        
        stratify_col = df['category'].copy()
        stratify_col[~df['category'].isin(stratifiable_classes)] = -1
        
        return train_test_split(
            df,
            test_size=test_size,
            random_state=42,
            stratify=stratify_col if len(stratify_col.unique()) > 1 else None
        )

    def _resample_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample data with proper handling of class imbalance."""
        resampled_dfs = []
        
        for category in df['category'].unique():
            category_df = df[df['category'] == category]
            n_samples = len(category_df)
            
            if n_samples > self.config.max_samples_per_class:
                sampled_df = category_df.sample(
                    n=self.config.max_samples_per_class,
                    random_state=42
                )
            elif n_samples < self.config.min_samples_per_class:
                sampled_df = category_df.sample(
                    n=self.config.min_samples_per_class,
                    replace=True,
                    random_state=42
                )
            else:
                sampled_df = category_df
                
            resampled_dfs.append(sampled_df)
        
        return pd.concat(resampled_dfs, ignore_index=True)

    def _tokenize_datasets(self, datasets: DatasetDict) -> DatasetDict:
        """Tokenize datasets with proper error handling."""
        try:
            # First rename the category column to labels
            datasets = DatasetDict({
                split: dataset.rename_column("category", "labels") 
                for split, dataset in datasets.items()
            })
            
            def tokenize_function(examples):
                return self.tokenizer(
                    examples["text"],
                    padding="max_length",
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt"
                )

            # Keep only the necessary columns during tokenization
            tokenized_datasets = datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=['text']  # Only remove the text column
            )

            return tokenized_datasets
        except Exception as e:
            logger.error(f"Error during tokenization: {str(e)}")
            raise

    def create_model(self) -> AutoModelForSequenceClassification:
        """Create and configure the model."""
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name,
                num_labels=len(self.label_encoder.classes_),
                ignore_mismatched_sizes=True
            )
            
            # Freeze specified layers
            for layer in model.roberta.encoder.layer[:self.config.frozen_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
            
            return model.to(self.device)
        except Exception as e:
            logger.error(f"Error creating model: {str(e)}")
            raise

    def train(self, tokenized_datasets: DatasetDict, class_weights: List[float]):
        """Train the model with proper error handling and logging."""
        try:
            # Calculate training steps
            train_size = len(tokenized_datasets["train"])
            effective_batch_size = self.config.batch_size * 4
            steps_per_epoch = (train_size + effective_batch_size - 1) // effective_batch_size
            
            training_args = TrainingArguments(
                output_dir="./results",
                evaluation_strategy="steps",
                eval_steps=steps_per_epoch // 2,
                save_strategy="steps",
                save_steps=steps_per_epoch // 2,
                logging_dir="./logs",
                logging_steps=10,
                load_best_model_at_end=True,
                metric_for_best_model="eval_f1_weighted",
                save_total_limit=2,
                learning_rate=self.config.learning_rate,
                max_grad_norm=self.config.max_grad_norm,
                num_train_epochs=self.config.num_epochs,
                weight_decay=self.config.weight_decay,
                per_device_train_batch_size=self.config.batch_size,
                gradient_accumulation_steps=4,
                warmup_ratio=self.config.warmup_ratio,
                no_cuda=(self.device.type == "cpu")
            )

            model = self.create_model()
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["validation"],
                data_collator=DataCollatorWithPadding(self.tokenizer),
                compute_metrics=self._create_compute_metrics(),
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
            )

            # Train and evaluate
            with mlflow.start_run():
                mlflow.log_params(training_args.to_dict())
                
                trainer.train()
                eval_results = trainer.evaluate()
                test_results = trainer.evaluate(tokenized_datasets["test"])
                
                # Log metrics
                # Sanitize and log metrics
                eval_metrics = {
                    "eval_loss": eval_results["eval_loss"],
                    "eval_accuracy": eval_results["eval_accuracy"],
                    "eval_f1_weighted": eval_results["eval_f1_weighted"]
                }
                
                # Add test metrics with sanitized names
                test_metrics = {
                    f"test_{k}": v for k, v in test_results.items()
                    if not k.startswith("f1_class_") or self._sanitize_metric_name(k) == k
                }
                
                mlflow.log_metrics({**eval_metrics, **test_metrics})
                
                # Save model and tokenizer
                model.save_pretrained("./results/model")
                self.tokenizer.save_pretrained("./results/model")
                
                logger.info("Training completed successfully")
                return eval_results, test_results

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

    def _sanitize_metric_name(self, name: str) -> str:
        """Sanitize metric names to be MLflow-compatible."""
        # Replace invalid characters with underscores
        sanitized = name.replace('/', '_').replace('(', '_').replace(')', '_')
        sanitized = ''.join(c if c.isalnum() or c in ['_', '-', '.', ' '] else '_' for c in sanitized)
        # Ensure the name doesn't start with a number
        if sanitized[0].isdigit():
            sanitized = 'n_' + sanitized
        return sanitized

    def _create_compute_metrics(self):
        """Create compute_metrics function with detailed class analysis."""
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            preds = np.argmax(predictions, axis=1)
            
            metrics = {
                "accuracy": accuracy_score(labels, preds),
                "precision": precision_score(labels, preds, average='weighted', zero_division=0),
                "f1_weighted": f1_score(labels, preds, average='weighted', zero_division=0)
            }
            
            # Get unique classes present in this evaluation
            unique_classes = np.unique(labels)
            present_class_names = [self.label_encoder.classes_[i] for i in unique_classes]
            
            # Calculate per-class metrics only for present classes
            report = classification_report(
                labels,
                preds,
                labels=unique_classes,  # Only use classes present in the data
                target_names=present_class_names,
                output_dict=True,
                zero_division=0
            )
            
            # Log metrics only for present classes
            for class_idx, class_name in zip(unique_classes, present_class_names):
                if class_name in report:
                    sanitized_name = self._sanitize_metric_name(class_name)
                    metrics[f"f1_class_{sanitized_name}"] = report[class_name]["f1-score"]
            
            # Log number of classes for debugging
            logger.info(f"Number of classes in evaluation: {len(unique_classes)}")
            logger.info(f"Classes present: {present_class_names}")
            
            return metrics
            
        return compute_metrics
