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
| Logistic Regression | 0.8859 | 0.9297 | 0.8872 | 0.8859 | 0.8852 | 0.7694 |
| Decision Tree | 0.788 | 0.7813 | 0.788 | 0.788 | 0.7868 | 0.5691 |
| kNN | 0.8859 | 0.936 | 0.8859 | 0.8859 | 0.8856 | 0.7686 |
| Naive Bayes | 0.913 | 0.9451 | 0.9134 | 0.913 | 0.9131 | 0.8246 |
| Random Forest (Ensemble) | 0.8696 | 0.9314 | 0.8694 | 0.8696 | 0.8694 | 0.7356 |
| XGBoost (Ensemble) | 0.8587 | 0.9219 | 0.8587 | 0.8587 | 0.8587 | 0.714 |

---

## Model Performance Observations

| ML Model Name | Observation about model performance |
|--------------|-------------------------------------|
| Logistic Regression | Strong baseline performance with **88.59% accuracy** and high **AUC (0.9297)**, indicating good class separability. Precision, recall, and F1 scores are well balanced, showing stable performance across classes. High MCC (0.7694) suggests reliable predictions, making it a solid interpretable model for medical risk prediction. |
| Decision Tree | Lowest performer among all models with **78.8% accuracy** and **AUC (0.7813)**, indicating weaker discrimination ability. Although highly interpretable, it may suffer from overfitting and limited generalization. Lower MCC (0.5691) reflects weaker balanced classification performance compared to other models. |
| kNN | Performs comparably to Logistic Regression with **88.59% accuracy** and strong **AUC (0.936)**. Balanced precision–recall metrics show consistent classification. Being distance-based, performance depends heavily on feature scaling and choice of *k*; current setup provides stable results. |
| Naive Bayes | **Best overall performer** with **91.3% accuracy**, highest **AUC (0.9451)**, and strongest **MCC (0.8246)**. Despite its feature independence assumption, it generalizes very well on this dataset. Excellent balance across precision, recall, and F1 makes it highly suitable for reliable predictions and real-time deployment. |
| Random Forest (Ensemble) | Good ensemble performance with **86.96% accuracy** and strong **AUC (0.9314)**. Improves over single Decision Tree by reducing variance and overfitting. Balanced metrics and moderate MCC (0.7356) indicate robust but slightly less optimal performance compared to top models. |
| XGBoost (Ensemble) | Reasonable performance with **85.87% accuracy** and **AUC (0.9219)**. Although gradient boosting typically performs strongly, here it trails behind Naive Bayes and kNN, possibly due to limited dataset size or hyperparameter tuning. MCC (0.714) indicates acceptable but not best class balance. |

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
    └── heart.csv
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

## Author

**Nagarjuna Manupati**  
ID: 2025AA052895

email:2025aa05895@wilp.bits-pilani.ac.in

---

