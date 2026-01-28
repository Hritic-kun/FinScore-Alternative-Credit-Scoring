import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, roc_curve

# 1. DATA SIMULATION (Mimicking Real Fintech Data)
# Moneyview looks at Income, Debt-to-Income (DTI), and employment tenure.
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

# Create a realistic "Default" target (Target = 1 if defaulted)
# Default probability increases with higher DTI and lower Bureau Score
logit = (0.05 * df['dti_ratio']) - (0.01 * df['bureau_score']) + (0.5 * df['inquiries_last_6m']) + 4
prob = 1 / (1 + np.exp(-logit))
df['default'] = (prob > np.percentile(prob, 85)).astype(int) # 15% default rate

# 2. FEATURE ENGINEERING (The "Fintech" Twist)
# Creating 'Alternative' metrics
df['installment_to_income'] = (df['loan_amount'] / 12) / (df['annual_income'] / 12)
df['risk_score_custom'] = df['bureau_score'] * (1 - (df['dti_ratio']/100))

# 3. TRAIN-TEST SPLIT
X = df.drop('default', axis=1)
y = df['default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. MODELING (XGBoost - Moneyview's JD requirement)
# We use scale_pos_weight to handle imbalanced default classes
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=5, # Crucial for 15% default rate
    objective='binary:logistic',
    random_state=42
)

model.fit(X_train, y_train)

# 5. EVALUATION & RESULTS
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

auc = roc_auc_score(y_test, y_pred_proba)
print(f"Model AUC-ROC Performance: {auc:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6. VISUALIZING RISK DRIVERS (Data Storytelling)
plt.figure(figsize=(10, 6))
xgb.plot_importance(model, importance_type='gain', ax=plt.gca())
plt.title("Key Risk Drivers for Moneyview Loans (Feature Importance)")
plt.show()

# 7. ROC CURVE
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f'XGBoost (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()
