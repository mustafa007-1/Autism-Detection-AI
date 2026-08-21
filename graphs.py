import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.preprocessing import OneHotEncoder
import joblib
import numpy as np

# ==========================================
# 1. SETUP & DATA PREPARATION
# ==========================================
print("Step 1: Loading data and recreating the test set...")

# Load the dataset
try:
    df = pd.read_csv("autism_final_processed.csv")
except FileNotFoundError:
    print("ERROR: 'autism_final_processed.csv' not found.")
    exit()

# Define the exact same features as used in training
features = ['a1_score', 'a2_score', 'a3_score', 'a4_score', 'a5_score',
            'a6_score', 'a7_score', 'a8_score', 'a9_score', 'a10_score',
            'age', 'gender', 'ethnicity', 'jaundice', 'autism', 'used_app_before']

X = df[features].copy()
y = df['asd_status']

# --- PREPROCESSING (MUST MATCH TRAIN.PY EXACTLY) ---
# Map Binary Columns
X['gender'] = X['gender'].map({'m': 1, 'f': 0})
X['jaundice'] = X['jaundice'].map({'yes': 1, 'no': 0})
X['autism'] = X['autism'].map({'yes': 1, 'no': 0})
X['used_app_before'] = X['used_app_before'].map({'yes': 1, 'no': 0})

# Encode Ethnicity
# We load the fitted encoder to ensure categories match perfectly
try:
    ohe = joblib.load('onehot_encoder.pkl')
    ethnicity_encoded = ohe.transform(X[['ethnicity']])
except FileNotFoundError:
    print("Warning: 'onehot_encoder.pkl' not found. Re-fitting (might differ slightly if data changed).")
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    ethnicity_encoded = ohe.fit_transform(X[['ethnicity']])

ethnicity_df = pd.DataFrame(ethnicity_encoded, columns=ohe.get_feature_names_out())

# Combine features
X = pd.concat([X.drop('ethnicity', axis=1).reset_index(drop=True), ethnicity_df], axis=1)

# Split Data (Using same random_state=42 ensures we get the EXACT same 'Test' set as train.py)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 2. LOAD MODEL
# ==========================================
print("Step 2: Loading the trained model...")
try:
    model = joblib.load('asd_model.pkl')
except FileNotFoundError:
    print("ERROR: 'asd_model.pkl' not found. Run 'train.py' first!")
    exit()

# Generate Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ==========================================
# 3. GENERATE PLOTS
# ==========================================
print("Step 3: Generating visualizations...")

# --- Plot A: Confusion Matrix ---
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['No ASD', 'ASD Traits'],
            yticklabels=['No ASD', 'ASD Traits'])
plt.title('Confusion Matrix: Performance on Test Data', fontsize=14)
plt.ylabel('Actual Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("✅ Created 'confusion_matrix.png'")
# plt.show() # Uncomment to see on screen

# --- Plot B: Feature Importance ---
plt.figure(figsize=(10, 8))
importances = model.feature_importances_
# Get feature names from the model inputs
feature_names = X.columns
feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_df = feat_df.sort_values(by='Importance', ascending=False).head(10) # Top 10

sns.barplot(x='Importance', y='Feature', data=feat_df, palette='viridis')
plt.title('Top 10 Features Driving the Prediction', fontsize=14)
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.savefig('feature_importance.png')
print("✅ Created 'feature_importance.png'")
# plt.show()

# --- Plot C: ROC Curve ---
plt.figure(figsize=(8, 6))
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve (Sensitivity vs False Alarm)', fontsize=14)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('roc_curve.png')
print("✅ Created 'roc_curve.png'")
# plt.show()

print("\nAll visualizations saved successfully!")