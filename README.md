# Machine Learning Classification Models - Assignment 2

## Problem Statement

This project implements and compares six different machine learning classification algorithms on a dataset to evaluate their performance using multiple evaluation metrics. The goal is to build an end-to-end machine learning pipeline including model training, evaluation, and deployment through an interactive web application.

---

## Dataset Description
Heart Failure Prediction Dataset

Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worldwide. Four out of 5CVD deaths are due to heart attacks and strokes, and one-third of these deaths occur prematurely in people under 70 years of age. Heart failure is a common event caused by CVDs and this dataset contains 11 features that can be used to predict a possible heart disease.

People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help.
### Dataset Source
**Dataset Name**: Heart Failure Prediction Dataset  
**Source**: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data

### Dataset Characteristics
- **Number of Instances**: 918 patient records
- **Number of Features**: 12 (11 input features (12 total columns including target))
- **Target Variable**: HeartDisease (0 = No heart disease, 1 = Heart disease)
- **Classification Type**: Binary Classification
- **Number of Classes**: 2 (Heart Disease / No Heart Disease)

### Features Description
Built by combining multiple heart-disease datasets into one standardized table.Final dataset size: 918 patients with 11 input features + 1 target variable.Mix of clinical measurements, test results, and demographic attributes.

Attribute Information:
- Age: age of the patient [years]
- Sex: sex of the patient [M: Male, F: Female]
- ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
- RestingBP: resting blood pressure [mm Hg]
- Cholesterol: serum cholesterol [mm/dl]
- FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
- RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
- MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]
- ExerciseAngina: exercise-induced angina [Y: Yes, N: No]
- Oldpeak: oldpeak = ST [Numeric value measured in depression]
- ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
- Target(HeartDisease): output class [1: heart disease, 0: Normal]

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
| Logistic Regression | 0.8671 | 0.9317 | 0.8672 | 0.8671 | 0.8667 | 0.7306 |
| Decision Tree | [0.XXXX] | [0.XXXX] | [0.XXXX] | [0.XXXX] | [0.XXXX] | [0.XXXX] |
| kNN | [0.XXXX] | [0.XXXX] | [0.XXXX] | [0.XXXX] | [0.XXXX] | [0.XXXX] |
| Naive Bayes | [0.XXXX] | [0.XXXX] | [0.XXXX] | [0.XXXX] | [0.XXXX] | [0.XXXX] |
| Random Forest (Ensemble) | [0.XXXX] | [0.XXXX] | [0.XXXX] | [0.XXXX] | [0.XXXX] | [0.XXXX] |
| XGBoost (Ensemble) | [0.XXXX] | [0.XXXX] | [0.XXXX] | [0.XXXX] | [0.XXXX] | [0.XXXX] |

---

## Model Performance Observations

| ML Model Name | Observation about model performance |
|--------------|-------------------------------------|
| Logistic Regression | Logistic Regression achieved accuracy 0.8671, AUC 0.9317, precision 0.8672, recall 0.8671, F1 0.8667 and MCC 0.7306. The model demonstrates strong discriminative ability (high AUC) with balanced precision and recall, indicating reliable overall performance. |
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
