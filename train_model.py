import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, classification_report
import joblib

# 1. Read cleaned data
df = pd.read_csv('heart.csv')

# 2. Separate Features (X) and Target (y)
target_col = 'output' if 'output' in df.columns else 'target'
X = df.drop(columns=[target_col])
y = df[target_col]

# 3. Train-Test Split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Evaluation
y_pred = model.predict(X_test_scaled)
print(f"Recall Score: {recall_score(y_test, y_pred) * 100:.2f}%")

# 7. Save Model & Scaler
joblib.dump(model, 'heart_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("✅ Success! 'heart_model.pkl' and 'scaler.pkl' generated.")