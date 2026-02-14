"""
Streamlit Web Application for ML Classification Models
Machine Learning Assignment 2
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="ML Classification Models",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #1C0770;
        padding: 10px;
        border-radius: 5px;
    }
    h1 {
        color: #1f77b4;
    }
    h2 {
        color: #ff7f0e;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("💡 Machine Learning Classification Models")
st.markdown("### Compare Multiple Classification Algorithms")
st.markdown("---")

# Sidebar
st.sidebar.header("📊 Configuration")
st.sidebar.markdown("Upload your test dataset and select a model to evaluate.")

# Model dictionary with descriptions
MODEL_INFO = {
    'Logistic Regression': {
        'file': 'model/logistic_regression_model.pkl',
        'description': 'Linear model for binary and multi-class classification'
    },
    'Decision Tree': {
        'file': 'model/decision_tree_model.pkl',
        'description': 'Tree-based model using recursive binary splitting'
    },
    'kNN': {
        'file': 'model/knn_model.pkl',
        'description': 'Instance-based learning using k-nearest neighbors'
    },
    'Naive Bayes': {
        'file': 'model/naive_bayes_model.pkl',
        'description': 'Probabilistic classifier based on Bayes theorem'
    },
    'Random Forest': {
        'file': 'model/random_forest_model.pkl',
        'description': 'Ensemble of decision trees using bagging'
    },
    'XGBoost': {
        'file': 'model/xgboost_model.pkl',
        'description': 'Gradient boosting ensemble method'
    }
}


@st.cache_resource
def load_model(model_name):
    """Load the trained model"""
    try:
        model_path = MODEL_INFO[model_name]['file']
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


@st.cache_resource
def load_preprocessors():
    """Load scaler and label encoder"""
    try:
        scaler = joblib.load('model/scaler.pkl')
        label_encoder = joblib.load('model/label_encoder.pkl')
        return scaler, label_encoder
    except Exception as e:
        st.error(f"Error loading preprocessors: {str(e)}")
        return None, None


def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate all evaluation metrics"""
    metrics = {}
    
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    metrics['Precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['Recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['F1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['MCC'] = matthews_corrcoef(y_true, y_pred)
    
    # AUC Score
    try:
        if y_pred_proba is not None:
            if len(np.unique(y_true)) == 2:
                metrics['AUC'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            else:
                metrics['AUC'] = roc_auc_score(y_true, y_pred_proba, 
                                               multi_class='ovr', average='weighted')
        else:
            metrics['AUC'] = 0.0
    except:
        metrics['AUC'] = 0.0
    
    return metrics


def plot_confusion_matrix(cm, classes):
    """Plot confusion matrix"""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    return fig


def main():
    """Main application function"""
    
    # Sidebar - File Upload
    st.sidebar.markdown("### 📁 Upload Test Data")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file", 
        type=['csv'],
        help="Upload your test dataset in CSV format"
    )
    
    # Sidebar - Model Selection
    st.sidebar.markdown("### 🎯 Select Model")
    selected_model = st.sidebar.selectbox(
        "Choose a classification model:",
        list(MODEL_INFO.keys()),
        help="Select which model to use for predictions"
    )
    
    # Display model description
    st.sidebar.markdown("#### Model Description")
    st.sidebar.info(MODEL_INFO[selected_model]['description'])
    
    # Main content area
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Dataset uploaded successfully! Shape: {df.shape}")
            
            # Show dataset preview
            with st.expander("📋 View Dataset Preview", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)
                st.markdown(f"**Total rows:** {df.shape[0]} | **Total columns:** {df.shape[1]}")
            
            # Target column selection
            st.markdown("### 🎯 Select Target Column")
            target_column = st.selectbox(
                "Choose the target column:",
                df.columns.tolist(),
                index=len(df.columns)-1  # Default to last column
            )
            
            if st.button("🚀 Run Predictions", type="primary", use_container_width=True):
                with st.spinner("Loading model and making predictions..."):
                    
                    # Separate features and target
                    X = df.drop(columns=[target_column])
                    y_true = df[target_column]
                    
                    # Load preprocessors
                    scaler, label_encoder = load_preprocessors()
                    
                    # Encode target if needed
                    if y_true.dtype == 'object':
                        try:
                            y_true_encoded = label_encoder.transform(y_true)
                        except:
                            # If label encoder fails, create a new one
                            from sklearn.preprocessing import LabelEncoder
                            temp_encoder = LabelEncoder()
                            y_true_encoded = temp_encoder.fit_transform(y_true)
                    else:
                        y_true_encoded = y_true.values
                    
                    # Handle categorical features
                    categorical_cols = X.select_dtypes(include=['object']).columns
                    if len(categorical_cols) > 0:
                        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
                    
                    # Scale features
                    if scaler is not None:
                        X_scaled = scaler.transform(X)
                    else:
                        X_scaled = X.values
                    
                    # Load model
                    model = load_model(selected_model)
                    
                    if model is not None:
                        # Make predictions
                        y_pred = model.predict(X_scaled)
                        
                        # Get probability predictions
                        try:
                            y_pred_proba = model.predict_proba(X_scaled)
                        except:
                            y_pred_proba = None
                        
                        # Calculate metrics
                        metrics = calculate_metrics(y_true_encoded, y_pred, y_pred_proba)
                        
                        # Display results
                        st.markdown("---")
                        st.markdown(f"## 📊 Evaluation Results: {selected_model}")
                        
                        # Metrics display in columns
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("🎯 Accuracy", f"{metrics['Accuracy']:.4f}")
                            st.metric("📈 Precision", f"{metrics['Precision']:.4f}")
                        
                        with col2:
                            st.metric("📉 AUC Score", f"{metrics['AUC']:.4f}")
                            st.metric("🔍 Recall", f"{metrics['Recall']:.4f}")
                        
                        with col3:
                            st.metric("⚖️ F1 Score", f"{metrics['F1']:.4f}")
                            st.metric("🔢 MCC Score", f"{metrics['MCC']:.4f}")
                        
                        st.markdown("---")
                        
                        # Confusion Matrix and Classification Report
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### 🎨 Confusion Matrix")
                            cm = confusion_matrix(y_true_encoded, y_pred)
                            classes = np.unique(y_true_encoded)
                            fig = plot_confusion_matrix(cm, classes)
                            st.pyplot(fig)
                        
                        with col2:
                            st.markdown("### 📋 Classification Report")
                            report = classification_report(y_true_encoded, y_pred, 
                                                          output_dict=True, zero_division=0)
                            report_df = pd.DataFrame(report).transpose()
                            st.dataframe(report_df.style.highlight_max(axis=0, 
                                                                       color='lightgreen'), 
                                        use_container_width=True)
                        
                        # Download predictions
                        st.markdown("---")
                        st.markdown("### 💾 Download Predictions")
                        
                        predictions_df = df.copy()
                        predictions_df['Predicted'] = y_pred
                        
                        csv = predictions_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions as CSV",
                            data=csv,
                            file_name=f"{selected_model.replace(' ', '_')}_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                        st.success("✅ Predictions completed successfully!")
                    
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure your CSV file is properly formatted and contains the required columns.")
    
    else:
        # Instructions when no file is uploaded
        st.info("👈 Please upload a CSV file to begin")
        
        st.markdown("""
        ### 📖 Instructions:
        
        1. **Upload Dataset**: Click on 'Browse files' in the sidebar to upload your test dataset (CSV format)
        2. **Select Target Column**: Choose the target variable from the dropdown
        3. **Choose Model**: Select one of the 6 classification models
        4. **Run Predictions**: Click the 'Run Predictions' button
        5. **View Results**: Analyze the evaluation metrics, confusion matrix, and classification report
        6. **Download**: Export predictions as CSV for further analysis
        
        ### 🎯 Available Models:
        """)
        
        for model_name, info in MODEL_INFO.items():
            st.markdown(f"- **{model_name}**: {info['description']}")
        
        st.markdown("""
        ### 📊 Evaluation Metrics:
        - **Accuracy**: Overall correctness of predictions
        - **AUC Score**: Area Under the ROC Curve
        - **Precision**: Positive predictive value
        - **Recall**: True positive rate (Sensitivity)
        - **F1 Score**: Harmonic mean of precision and recall
        - **MCC Score**: Matthews Correlation Coefficient
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666;'>
            <p>Machine Learning Assignment 2 | M.Tech (AIML/DSE) | BITS Pilani</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
