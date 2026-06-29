# Diabetes Prediction using SVM

Predict whether a patient has diabetes based on diagnostic measurements using a Support Vector Machine (SVM) classifier. This project includes both a Jupyter notebook for model development/training and a deployed web application for predictions.

---

## Dataset

**File:** `diabetes.csv`
- Source: Pima Indians Diabetes Database (Kaggle / UCI ML Repository)
- 768 samples, 8 features + 1 target variable
- Target: `Outcome` (0 = Non-diabetic, 1 = Diabetic)

### Features
| Column | Description |
|--------|-------------|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration (2 hours after oral glucose tolerance test) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-Hour serum insulin (mu U/ml) |
| BMI | Body mass index (weight in kg/(height in m²)) |
| DiabetesPedigreeFunction | Likelihood of diabetes based on family history |
| Age | Age in years |

---

## Exploratory Data Analysis (EDA)

**File:** `info.ipynb`

### Key Steps:

1. **Data Loading**
   - Libraries: `pandas`, `numpy`
   - Load `diabetes.csv` into a DataFrame for analysis

2. **Data Inspection**
   - Check shapes, data types, and first few rows
   - Identify missing values or zero values (imputed in preprocessing)

3. **Statistical Summary**
   - `df.describe()`: mean, std, min, max, quartiles for each feature
   - Group by `Outcome` to see differences between diabetic and non-diabetic groups

4. **Data Visualization**
   - Count plots for outcome distribution (class balance check)
   - Correlation heatmaps to identify multicollinearity
   - Pair plots / scatter plots to visualize feature relationships

5. **Data Cleaning / Preprocessing**
   - Replace biologically implausible zeros with NaN
   - Impute missing values (mean/median) or remove rows
   - Normalize data for SVM (features with different scales)

---

## Feature Engineering & Preprocessing

### Standardization
SVM is sensitive to feature scales. We use `StandardScaler` from scikit-learn:

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### Train-Test Split
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)
```
- `stratify=y`: ensures same class distribution in train and test sets
- `random_state=2`: reproducibility

---

## Model: Support Vector Machine (SVM)

### What is SVM?
SVM is a supervised learning algorithm used for classification and regression. For classification, it finds the optimal hyperplane that maximizes the margin between classes in feature space.

- **Goal:** Find a decision boundary that separates two classes with the widest possible margin
- **Support Vectors:** Data points closest to the decision boundary (they define the margin)
- **Kernel Trick:** Maps non-linear data into higher dimensions where it becomes linearly separable

### Why SVM for this dataset?
- Works well with high-dimensional, small dataset
- Robust to overfitting (especially with regularization parameter C)
- Effective when classes are separable

---

## Kernel Selection

### Kernel Used: `linear`

```python
from sklearn import svm
classifier = svm.SVC(kernel='linear', C=1.0)
classifier.fit(X_train, y_train)
```

### Why Linear Kernel?
1. **Interpretability:** Coefficients directly show feature importance
2. **Speed:** Faster training than RBF or polynomial on small datasets
3. **Performance:** The Pima Indian dataset is approximately linearly separable; linear kernel gave the best test accuracy
4. **Avoids Overfitting:** Non-linear kernels (RBF) may overfit on small datasets

### Other Kernels Tested (if any):
- **RBF (Radial Basis Function):** Good for non-linear boundaries; risk of overfitting
- **Polynomial:** Captures polynomial relationships; more parameters to tune
- **Sigmoid:** Less common for binary classification

---

## Model Evaluation

### Metrics Used:
```python
from sklearn.metrics import accuracy_score

train_acc = accuracy_score(classifier.predict(X_train), y_train)
test_acc = accuracy_score(classifier.predict(X_test), y_test)
```

### Results Achieved:
- **Training Accuracy:** ~83.12%
- **Test Accuracy:** ~77.27%

### Interpretation:
- The gap between training and test accuracy is small → model generalizes well
- 77.27% test accuracy is decent for a simple linear model on this dataset
- Further improvements could be achieved with:
  - More data
  - Feature engineering (e.g., BMI categories, age groups)
  - Hyperparameter tuning (GridSearchCV on C and gamma for RBF)
  - Ensemble methods (Random Forest, XGBoost)

---

## Deployment Files

### 1. Streamlit Application (`app.py`)
- **Framework:** Streamlit
- **Purpose:** Interactive web app for real-time predictions
- **Features:**
  - User input form for all 8 health metrics
  - Real-time prediction with probability display
  - Cached model training and scaler for performance
  - Displays training/test accuracy

### 2. Static HTML/JS Application (`index.html`) + (`model_params.json`)
- **Framework:** Pure HTML/CSS/JavaScript (no server required)
- **Purpose:** Deploy to GitHub Pages as a static site
- **How it works:**
  - Model parameters (scaler means, scales, and SVM coefficients) are stored in JSON
  - JavaScript replicates the Python `StandardScaler` and linear SVM prediction
  - Prediction happens client-side in the browser
- **Limitation:** Cannot retrain; model is static

### 3. Model Parameters (`model_params.json`)
```json
{
  "scaler": {
    "mean": [...],   // mean values for each feature for standardization
    "scale": [...]   // std values for each feature for standardization
  },
  "classifier": {
    "coef": [[...]],  // SVM weights (8 coefficients for 8 features)
    "intercept": [...] // SVM bias term
  }
}
```

These parameters allow the JavaScript app to replicate the trained Python model.

---

## Project Structure

```
diabetes-predictor/
├── diabetes.csv                 # Dataset
├── info.ipynb                   # Jupyter notebook: EDA, preprocessing, model training
├── diabetes_svm_explained.html  # Static HTML report (EDA + SVM explanation)
├── model_params.json            # Exported model parameters (for static deployment)
├── index.html                   # Static web app (deploys on GitHub Pages)
├── app.py                       # Streamlit app (for local/server deployment)
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

---

## Running Locally

### Streamlit App
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Static HTML
Open `index.html` in any web browser (no server needed).

---

## Deployment

### GitHub Pages (Static)
- Enabled in repo Settings → Pages → Source: `main` branch
- Live at: `https://yuv9799.github.io/diabetes-predictor/`

### Render / Streamlit Community Cloud (Dynamic)
- Connect GitHub repo
- Build command: `pip install -r requirements.txt`
- Start command: `streamlit run app.py --server.port $PORT --server.headless true`

---

## Future Improvements
- Hyperparameter tuning with `GridSearchCV` or `RandomizedSearchCV`
- Try advanced models: Random Forest, XGBoost, Neural Networks
- Add SHAP/LIME plots for model explainability
- Collect more data to improve accuracy
- Add user authentication and prediction history (for full-stack version)

---
**Built with:** Python, Pandas, NumPy, Scikit-learn, Streamlit, HTML/JS