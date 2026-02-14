# Quick Commands Reference
## ML Assignment 2 - Essential Commands

---

## Setup Commands

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Install with Break System Packages (if needed)
```bash
pip install -r requirements.txt --break-system-packages
```

---

## Running the Project

### Train Models (Option 1 - Python Script)
```bash
cd model
python train_models.py
```

### Train Models (Option 2 - Jupyter Notebook)
```bash
jupyter notebook model/ML_Assignment_Training.ipynb
```

### Run Streamlit App Locally
```bash
streamlit run app.py
```

### Run Streamlit on Specific Port
```bash
streamlit run app.py --server.port 8502
```

---

## Git Commands

### Initialize Repository
```bash
git init
git add .
git commit -m "Initial commit: ML Classification Assignment"
```

### Connect to GitHub
```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git branch -M main
git push -u origin main
```

### Update After Changes
```bash
git add .
git commit -m "Updated: [describe your changes]"
git push
```

### Check Status
```bash
git status
```

### View Commit History
```bash
git log --oneline
```

---

## Useful Python Commands

### Check Installed Packages
```bash
pip list
```

### Check Specific Package Version
```bash
pip show scikit-learn
```

### Create requirements.txt from Environment
```bash
pip freeze > requirements.txt
```

### Check Python Version
```bash
python --version
```

---

## File Management Commands

### List Files
```bash
ls -la
```

### Create Directory
```bash
mkdir model
```

### Copy Files
```bash
cp source.csv destination.csv
```

### Move Files
```bash
mv old_location.csv new_location.csv
```

### Remove Files (Be Careful!)
```bash
rm filename.txt
```

---

## Jupyter Notebook Commands

### Start Jupyter
```bash
jupyter notebook
```

### Start on Specific Port
```bash
jupyter notebook --port=8889
```

### Convert Notebook to Python
```bash
jupyter nbconvert --to python notebook.ipynb
```

---

## Debugging Commands

### Check if Port is in Use
```bash
# Linux/Mac
lsof -i :8501

# Windows
netstat -ano | findstr :8501
```

### Kill Process on Port (if needed)
```bash
# Linux/Mac
kill -9 <PID>

# Windows
taskkill /PID <PID> /F
```

### Test Python Import
```python
python -c "import streamlit; print(streamlit.__version__)"
python -c "import sklearn; print(sklearn.__version__)"
python -c "import xgboost; print(xgboost.__version__)"
```

---

## Streamlit Specific Commands

### Clear Cache
```bash
streamlit cache clear
```

### Run with Debugging
```bash
streamlit run app.py --logger.level=debug
```

### Check Streamlit Version
```bash
streamlit --version
```

---

## Dataset Inspection Commands

### Quick View CSV
```bash
head -n 10 dataset.csv
```

### Count Lines
```bash
wc -l dataset.csv
```

### Python Quick Inspection
```python
import pandas as pd
df = pd.read_csv('dataset.csv')
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(df.head())
```

---

## Model File Commands

### Check Model File Size
```bash
ls -lh model/*.pkl
```

### Count Model Files
```bash
ls -1 model/*.pkl | wc -l
```

---

## Common Error Fixes

### ModuleNotFoundError
```bash
pip install <module_name>
```

### Permission Denied
```bash
chmod +x script.py
```

### Jupyter Kernel Issues
```bash
python -m ipykernel install --user --name myenv
```

---

## Streamlit Cloud Deployment

### From Web Interface
1. Go to https://streamlit.io/cloud
2. Sign in with GitHub
3. Click "New app"
4. Select repository → branch → app.py
5. Click "Deploy"

### Check Deployment Status
- View logs in Streamlit Cloud dashboard
- Look for error messages
- Verify all dependencies installed

---

## Verification Commands

### Verify All Required Files Exist
```bash
ls app.py requirements.txt README.md
ls model/train_models.py
ls model/*.pkl
```

### Count Model Files (Should be 8)
```bash
# 6 model files + scaler + label_encoder = 8
ls model/*.pkl | wc -l
```

### Verify README Completeness
```bash
grep -c "ML Model Name" README.md  # Should find your table
grep -c "Observation" README.md    # Should find observations section
```

---

## Testing Commands

### Test Streamlit App Components
```python
# In Python
import joblib

# Test model loading
model = joblib.load('model/logistic_regression_model.pkl')
print("Model loaded successfully")

# Test scaler
scaler = joblib.load('model/scaler.pkl')
print("Scaler loaded successfully")
```

### Quick Test All Models Load
```python
import joblib
import os

model_files = [f for f in os.listdir('model') if f.endswith('.pkl')]
for mf in model_files:
    try:
        joblib.load(f'model/{mf}')
        print(f"✓ {mf}")
    except Exception as e:
        print(f"✗ {mf}: {e}")
```

---

## Emergency Commands

### Kill All Python Processes
```bash
# Linux/Mac
pkill -9 python

# Windows
taskkill /F /IM python.exe
```

### Reinstall All Packages
```bash
pip uninstall -r requirements.txt -y
pip install -r requirements.txt
```

---

## Submission Checklist Commands

### Generate File Tree
```bash
tree -L 2
# Or if tree not installed:
find . -maxdepth 2 -type f
```

### Count Total Lines of Code
```bash
find . -name "*.py" | xargs wc -l
```

---

## Quick Reminders

✓ Always activate virtual environment before installing packages  
✓ Commit frequently to GitHub  
✓ Test locally before deploying  
✓ Take BITS Lab screenshot before finishing  
✓ Verify all links work before submitting PDF  
✓ Submit (not save as draft) on Taxila  
✓ Deadline: 15-Feb-2026 23:59 PM  

---

## One-Line Complete Setup

```bash
pip install -r requirements.txt && cd model && python train_models.py && cd .. && streamlit run app.py
```

---

## For Copy-Paste: Update Training Script

```python
# Update these lines in model/train_models.py or notebook
DATA_PATH = 'your_dataset.csv'  # ← CHANGE THIS
TARGET_COLUMN = 'target'         # ← CHANGE THIS
```

---

## Screenshot Reminder

📸 **DON'T FORGET**: Take screenshot showing execution on BITS Virtual Lab!
This is worth 1 mark!

---

**Good luck with your assignment! 🚀**
