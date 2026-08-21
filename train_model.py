import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import joblib

# --- 1. Load Data ---
print("Step 1: Loading data...")
try:
    df = pd.read_csv("autism_final_processed.csv")
except FileNotFoundError:
    print("ERROR: 'autism_final_processed.csv' not found.")
    exit()

# --- 2. Preprocessing (Turning Data into Numbers) ---
print("Step 2: Preparing data...")

features = ['a1_score', 'a2_score', 'a3_score', 'a4_score', 'a5_score',
            'a6_score', 'a7_score', 'a8_score', 'a9_score', 'a10_score',
            'age', 'gender', 'ethnicity', 'jaundice', 'autism', 'used_app_before']

X = df[features].copy()
y = df['asd_status']

# Map Binary Columns (Yes/No -> 1/0)
X['gender'] = X['gender'].map({'m': 1, 'f': 0})
X['jaundice'] = X['jaundice'].map({'yes': 1, 'no': 0})
X['autism'] = X['autism'].map({'yes': 1, 'no': 0})
X['used_app_before'] = X['used_app_before'].map({'yes': 1, 'no': 0})

# Encode Ethnicity (One-Hot Encoding)
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ethnicity_encoded = ohe.fit_transform(X[['ethnicity']])
ethnicity_df = pd.DataFrame(ethnicity_encoded, columns=ohe.get_feature_names_out())

# Combine everything
X = pd.concat([X.drop('ethnicity', axis=1).reset_index(drop=True), ethnicity_df], axis=1)

# --- 3. Split Data ---
# 80% for Training (Studying), 20% for Testing (The Exam)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. Train Model ---
print("Step 3: Training the Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- 5. Evaluate (The Report Card) ---
print("\n" + "="*40)
print("       MODEL EVALUATION METRICS       ")
print("="*40)

y_pred = model.predict(X_test)

# A. Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {acc:.2%}")
print("(This means the model is correct {:.0f}% of the time)".format(acc*100))

# B. Classification Report (Precision, Recall, F1)
print("\n📊 Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No ASD', 'ASD Traits']))

# C. Confusion Matrix (The Raw Numbers)
cm = confusion_matrix(y_test, y_pred)
print("\n🔢 Confusion Matrix:")
print(f"True Negatives (Correctly said No): {cm[0][0]}")
print(f"False Positives (Wrongly said Yes): {cm[0][1]}")
print(f"False Negatives (Wrongly said No):  {cm[1][0]}")
print(f"True Positives  (Correctly said Yes): {cm[1][1]}")
print("="*40 + "\n")

# --- 6. Save ---
joblib.dump(model, 'asd_model.pkl')
joblib.dump(ohe, 'onehot_encoder.pkl')
joblib.dump(features, 'features.pkl')
print("Model saved successfully! Ready for app.py")