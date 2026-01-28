import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, roc_curve

# 1. DATA SIMULATION (Aligned with README "Alternative Data" Focus)
np.random.seed(42)
n_samples = 5000

data = {
    'annual_income': np.random.normal(600000, 200000, n_samples), # In INR
    'dti_ratio': np.random.uniform(10, 50, n_samples),           # Debt-to-Income
    'emp_length_years': np.random.randint(1, 15, n_samples),
    'inquiries_last_6m': np.random.poisson(1, n_samples),
    'bureau_score': np.random.normal(700, 50, n_samples),        # Traditional score
    'loan_amount': np.random.uniform(50000, 500000, n_samples)
}

df = pd.DataFrame(data)

# 2. TARGET MAPPING (Logic: Map Default/Charged-Off to 1)
# Default probability modeled as a function of DTI and low Bureau Scores
logit = (0.05 * df['dti_ratio']) - (0.01 * df['bureau_score']) + (0.5 * df['inquiries_last_6m']) + 4
prob = 1 / (1 + np.exp(-logit))
df['default'] = (prob > np.percentile(prob, 85)).astype(int) # Simulated 15% default rate

# 3. FEATURE ENGINEERING (Fintech-specific "Alternative" Metrics)
# Repayment capacity: how much of monthly income goes to the loan?
df['installment_to_income'] = (df['loan_amount'] / 12) / (df['annual_income'] / 12)

# Custom Risk Score: Blending traditional score with DTI health
df['custom_risk_score'] = df['bureau_score'] * (1 - (df['dti_ratio']/100))

# 4. TRAIN-TEST SPLIT
X = df.drop('default', axis=1)
y = df['default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. MODELING (Using Regularized XGBoost as per README)
# scale_pos_weight is the ratio of negative to positive samples (approx 5:1)
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=5.6, # Specifically handles the 15% imbalanced default rate
    objective='binary:logistic',
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# 6. EVALUATION & RESULTS
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

auc = roc_auc_score(y_test, y_pred_proba)
print(f"Model AUC-ROC Performance: {auc:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. VISUALIZING RISK DRIVERS (Data Storytelling for Recruiters)

plt.figure(figsize=(10, 6))
xgb.plot_importance(model, importance_type='gain', ax=plt.gca(), color='skyblue')
plt.title("Key Risk Drivers: Feature Importance (Gain)")
plt.show()

# 8. ROC CURVE (Visualizing Discriminative Power)

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f'XGBoost Classifier (AUC = {auc:.2f})', color='darkorange', lw=2)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Credit Risk Model - ROC Curve')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()
