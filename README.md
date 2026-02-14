# Machine Learning Classification Models - Assignment 2

## Problem Statement

This project implements and compares six different machine learning classification algorithms on a dataset to evaluate their performance using multiple evaluation metrics. The goal is to build an end-to-end machine learning pipeline including model training, evaluation, and deployment through an interactive web application.

---

## Dataset Description

### Dataset Source
**Dataset Name**: [Your Dataset Name Here]  
**Source**: [Kaggle/UCI Repository Link]

### Dataset Characteristics
- **Number of Instances**: [e.g., 1000]
- **Number of Features**: [e.g., 15]
- **Target Variable**: [e.g., Class/Label]
- **Classification Type**: [Binary/Multi-class]
- **Number of Classes**: [e.g., 2 or 5]

### Features Description
[Provide a brief description of the key features in your dataset]

Example:
- Feature 1: [Description]
- Feature 2: [Description]
- ...
- Target: [Description]

### Data Preprocessing
- Handled missing values (if any)
- Encoded categorical variables using one-hot encoding
- Scaled numerical features using StandardScaler
- Split ratio: 80% training, 20% testing
- Stratified split to maintain class distribution

---

## Models Used

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|-------|-----|
| Logistic Regression | [0.XXXX] | [0.XXXX] | [0.XXXX] | [0.XXXX] | [0.XXXX] | [0.XXXX] |
| Decision Tree | [0.XXXX] | [0.XXXX] | [0.XXXX] | [0.XXXX] | [0.XXXX] | [0.XXXX] |
| kNN | [0.XXXX] | [0.XXXX] | [0.XXXX] | [0.XXXX] | [0.XXXX] | [0.XXXX] |
| Naive Bayes | [0.XXXX] | [0.XXXX] | [0.XXXX] | [0.XXXX] | [0.XXXX] | [0.XXXX] |
| Random Forest (Ensemble) | [0.XXXX] | [0.XXXX] | [0.XXXX] | [0.XXXX] | [0.XXXX] | [0.XXXX] |
| XGBoost (Ensemble) | [0.XXXX] | [0.XXXX] | [0.XXXX] | [0.XXXX] | [0.XXXX] | [0.XXXX] |

---

## Model Performance Observations

| ML Model Name | Observation about model performance |
|--------------|-------------------------------------|
| Logistic Regression | [Your observation: e.g., "Logistic Regression showed moderate performance with an accuracy of X%. It performed well on linearly separable classes but struggled with complex decision boundaries. The model demonstrated good recall but lower precision, indicating more false positives."] |
| Decision Tree | [Your observation: e.g., "Decision Tree achieved an accuracy of X% with tendency to overfit on training data. The model showed high variance and was sensitive to small changes in the dataset. Performance could be improved with proper pruning."] |
| kNN | [Your observation: e.g., "K-Nearest Neighbors with k=5 achieved X% accuracy. The model performed well but was computationally expensive for predictions. Performance was sensitive to the choice of k and the distance metric used."] |
| Naive Bayes | [Your observation: e.g., "Gaussian Naive Bayes showed X% accuracy. Despite the strong independence assumption, it performed reasonably well and was the fastest to train. However, it had lower precision compared to other models."] |
| Random Forest (Ensemble) | [Your observation: e.g., "Random Forest achieved the second-best performance with X% accuracy and excellent AUC score of X. The ensemble approach reduced overfitting compared to a single decision tree and provided robust predictions with good generalization."] |
| XGBoost (Ensemble) | [Your observation: e.g., "XGBoost demonstrated the best overall performance with X% accuracy and the highest F1 score of X. The gradient boosting technique effectively handled the dataset complexity and achieved superior results across all metrics including the highest MCC score."] |

---

## Repository Structure

```
ml_assignment_2/
│
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── model/                          # Model directory
│   ├── train_models.py            # Training script for all models
│   ├── logistic_regression_model.pkl
│   ├── decision_tree_model.pkl
│   ├── knn_model.pkl
│   ├── naive_bayes_model.pkl
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   ├── scaler.pkl                 # StandardScaler for feature scaling
│   └── label_encoder.pkl          # LabelEncoder for target encoding
│
└── data/                           # (Optional) Dataset directory
    └── your_dataset.csv
```

---

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Local Installation

1. **Clone the repository**
```bash
git clone <your-repository-url>
cd ml_assignment2
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Train the models** (if not already trained)
```bash
cd model
python train_models.py
```

4. **Run the Streamlit app**
```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

---

## Using the Web Application

### Features

1. **Dataset Upload** (CSV format)
   - Upload your test dataset through the sidebar
   - Preview the dataset before making predictions

2. **Model Selection**
   - Choose from 6 different classification models
   - View model descriptions and characteristics

3. **Evaluation Metrics Display**
   - Accuracy, AUC, Precision, Recall, F1 Score, MCC
   - Visual representation of model performance

4. **Confusion Matrix**
   - Interactive confusion matrix visualization
   - Identify misclassification patterns

5. **Classification Report**
   - Detailed per-class performance metrics
   - Precision, recall, and F1-score for each class

6. **Download Predictions**
   - Export predictions as CSV file
   - Includes original data with predicted labels

### Step-by-Step Usage

1. **Upload Dataset**: Click "Browse files" in the sidebar and select your CSV file
2. **Select Target Column**: Choose the column containing true labels
3. **Choose Model**: Select one of the 6 classification models from the dropdown
4. **Run Predictions**: Click the "Run Predictions" button
5. **Analyze Results**: Review metrics, confusion matrix, and classification report
6. **Download**: Export predictions for further analysis

---

## Deployment on Streamlit Community Cloud

### Deployment Steps

1. **Push to GitHub**
   - Ensure all files are committed and pushed to your GitHub repository

2. **Visit Streamlit Cloud**
   - Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
   - Sign in with your GitHub account

3. **Deploy New App**
   - Click "New App"
   - Select your repository
   - Choose the branch (usually `main`)
   - Set the main file path: `app.py`
   - Click "Deploy"

4. **Wait for Deployment**
   - The app will be built and deployed (usually takes 2-5 minutes)
   - You'll receive a public URL for your app

### Live App Link
🔗 **[Your Streamlit App URL will be here]**

---

## Technologies Used

- **Python 3.8+**: Programming language
- **Scikit-learn**: Machine learning library
- **XGBoost**: Gradient boosting framework
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib & Seaborn**: Data visualization
- **Joblib**: Model serialization

---

## Model Details

### 1. Logistic Regression
- Linear model for classification
- Uses sigmoid function for probability estimation
- Hyperparameters: max_iter=1000, random_state=42

### 2. Decision Tree Classifier
- Tree-based model using recursive partitioning
- Non-parametric supervised learning
- Hyperparameters: random_state=42

### 3. K-Nearest Neighbors (kNN)
- Instance-based learning algorithm
- Classifies based on majority vote of k neighbors
- Hyperparameters: n_neighbors=5

### 4. Naive Bayes (Gaussian)
- Probabilistic classifier based on Bayes' theorem
- Assumes feature independence
- Uses Gaussian distribution for continuous features

### 5. Random Forest (Ensemble)
- Ensemble of decision trees
- Uses bagging and random feature selection
- Hyperparameters: n_estimators=100, random_state=42

### 6. XGBoost (Ensemble)
- Gradient boosting decision trees
- Sequential ensemble learning
- Hyperparameters: random_state=42, eval_metric='logloss'

---

## Evaluation Metrics Explained

- **Accuracy**: Proportion of correct predictions (TP+TN)/(TP+TN+FP+FN)
- **AUC Score**: Area Under the ROC Curve - model's ability to distinguish between classes
- **Precision**: Proportion of positive identifications that were actually correct (TP)/(TP+FP)
- **Recall**: Proportion of actual positives correctly identified (TP)/(TP+FN)
- **F1 Score**: Harmonic mean of precision and recall
- **MCC Score**: Matthews Correlation Coefficient - balanced measure considering all confusion matrix elements

---

## Future Improvements

- Implement hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- Add feature importance visualization
- Include ROC curves and PR curves
- Implement cross-validation for more robust evaluation
- Add support for custom datasets with automatic preprocessing
- Include model interpretation tools (SHAP, LIME)

---

## Author

**[Your Name]**  
M.Tech (AIML/DSE)  
BITS Pilani - Work Integrated Learning Programme

---

## Acknowledgments

- BITS Pilani for the assignment guidelines
- Scikit-learn and XGBoost documentation
- Streamlit community for deployment resources

---

## License

This project is created for academic purposes as part of the Machine Learning course assignment.

---

## Contact

For any queries regarding this project:
- Email: [your.email@example.com]
- GitHub: [your-github-username]

---

**Submission Date**: 15-Feb-2026  
**Course**: Machine Learning  
**Assignment**: Assignment 2
