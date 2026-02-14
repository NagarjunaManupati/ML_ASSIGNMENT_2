"""
Machine Learning Assignment 2
Multiple Classification Models Implementation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)
import joblib
import warnings
warnings.filterwarnings('ignore')

class MLClassifierPipeline:
    """
    Pipeline for training and evaluating multiple classification models
    """
    
    def __init__(self, data_path):
        """
        Initialize the pipeline with dataset path
        """
        self.data_path = data_path
        self.models = {}
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_and_prepare_data(self, target_column):
        """
        Load dataset and prepare train-test split
        """
        print("Loading dataset...")
        df = pd.read_csv(self.data_path)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Features: {df.shape[1] - 1}, Instances: {df.shape[0]}")
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Encode target if categorical
        if y.dtype == 'object':
            y = self.label_encoder.fit_transform(y)
        
        # Handle categorical features
        categorical_cols = X.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Test set size: {self.X_test.shape[0]}")
        print(f"Number of classes: {len(np.unique(y))}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def initialize_models(self):
        """
        Initialize all 6 classification models
        """
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'kNN': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB(),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
        }
        
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Calculate all required evaluation metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['Accuracy'] = accuracy_score(y_true, y_pred)
        metrics['Precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['Recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['F1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['MCC'] = matthews_corrcoef(y_true, y_pred)
        
        # AUC Score
        try:
            if y_pred_proba is not None:
                if len(np.unique(y_true)) == 2:
                    # Binary classification
                    metrics['AUC'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:
                    # Multi-class classification
                    metrics['AUC'] = roc_auc_score(y_true, y_pred_proba, 
                                                   multi_class='ovr', average='weighted')
            else:
                metrics['AUC'] = 0.0
        except:
            metrics['AUC'] = 0.0
        
        return metrics
    
    def train_and_evaluate(self):
        """
        Train all models and evaluate performance
        """
        print("\n" + "="*80)
        print("TRAINING AND EVALUATING MODELS")
        print("="*80)
        
        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")
            
            # Train model
            model.fit(self.X_train, self.y_train)
            
            # Predictions
            y_pred = model.predict(self.X_test)
            
            # Get probability predictions if available
            try:
                y_pred_proba = model.predict_proba(self.X_test)
            except:
                y_pred_proba = None
            
            # Calculate metrics
            metrics = self.calculate_metrics(self.y_test, y_pred, y_pred_proba)
            
            # Store results
            self.results[model_name] = {
                'model': model,
                'metrics': metrics,
                'predictions': y_pred,
                'confusion_matrix': confusion_matrix(self.y_test, y_pred),
                'classification_report': classification_report(self.y_test, y_pred)
            }
            
            print(f"✓ {model_name} trained successfully")
            print(f"  Accuracy: {metrics['Accuracy']:.4f}")
            print(f"  AUC: {metrics['AUC']:.4f}")
            print(f"  F1 Score: {metrics['F1']:.4f}")
        
        print("\n" + "="*80)
        print("TRAINING COMPLETED")
        print("="*80)
    
    def save_models(self, save_dir='model'):
        """
        Save all trained models
        """
        print("\nSaving models...")
        for model_name, result in self.results.items():
            model_filename = f"{save_dir}/{model_name.replace(' ', '_').lower()}_model.pkl"
            joblib.dump(result['model'], model_filename)
            print(f"✓ Saved {model_name}")
        
        # Save scaler and label encoder
        joblib.dump(self.scaler, f"{save_dir}/scaler.pkl")
        joblib.dump(self.label_encoder, f"{save_dir}/label_encoder.pkl")
        print("✓ Saved scaler and label encoder")
    
    def print_results_table(self):
        """
        Print comparison table of all models
        """
        print("\n" + "="*100)
        print("MODEL COMPARISON TABLE")
        print("="*100)
        
        print(f"{'ML Model Name':<25} {'Accuracy':<10} {'AUC':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'MCC':<10}")
        print("-"*100)
        
        for model_name, result in self.results.items():
            metrics = result['metrics']
            print(f"{model_name:<25} "
                  f"{metrics['Accuracy']:<10.4f} "
                  f"{metrics['AUC']:<10.4f} "
                  f"{metrics['Precision']:<10.4f} "
                  f"{metrics['Recall']:<10.4f} "
                  f"{metrics['F1']:<10.4f} "
                  f"{metrics['MCC']:<10.4f}")
        
        print("="*100)
    
    def export_results_to_csv(self, filename='model_results.csv'):
        """
        Export results to CSV for easy inclusion in README
        """
        results_data = []
        for model_name, result in self.results.items():
            metrics = result['metrics']
            results_data.append({
                'ML Model Name': model_name,
                'Accuracy': f"{metrics['Accuracy']:.4f}",
                'AUC': f"{metrics['AUC']:.4f}",
                'Precision': f"{metrics['Precision']:.4f}",
                'Recall': f"{metrics['Recall']:.4f}",
                'F1': f"{metrics['F1']:.4f}",
                'MCC': f"{metrics['MCC']:.4f}"
            })
        
        df_results = pd.DataFrame(results_data)
        df_results.to_csv(filename, index=False)
        print(f"\n✓ Results exported to {filename}")
        return df_results


def main():
    """
    Main execution function
    """
    # Example usage - You need to update these values based on your dataset
    DATA_PATH = 'your_dataset.csv'  # Update this
    TARGET_COLUMN = 'target'  # Update this with your target column name
    
    # Initialize pipeline
    pipeline = MLClassifierPipeline(DATA_PATH)
    
    # Load and prepare data
    pipeline.load_and_prepare_data(TARGET_COLUMN)
    
    # Initialize models
    pipeline.initialize_models()
    
    # Train and evaluate
    pipeline.train_and_evaluate()
    
    # Print results
    pipeline.print_results_table()
    
    # Save models
    pipeline.save_models()
    
    # Export results
    pipeline.export_results_to_csv()
    
    print("\n✓ All tasks completed successfully!")


if __name__ == "__main__":
    main()
