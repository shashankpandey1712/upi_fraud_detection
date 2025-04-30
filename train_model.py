import pandas as pd
import os

print("Step 1: Checking file...")
if os.path.exists("creditcard.csv"):
    print("✅ File found")
else:
    print("❌ File not found")
    exit()

print("Step 2: Loading dataset...")
try:
    df = pd.read_csv("creditcard.csv")
    print("✅ Dataset loaded")
except Exception as e:
    print(f"❌ Failed to load dataset: {e}")
    exit()

print("Step 3: Checking dataset shape...")
print(df.shape)

print("Step 4: Splitting data...")
try:
    X = df.drop("Class", axis=1)
    y = df["Class"]
    print("✅ Data split into X and y")
except Exception as e:
    print(f"❌ Error in splitting data: {e}")
    exit()

print("Step 5: Importing ML modules...")
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

print("Step 6: Splitting train/test...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("✅ Data split complete")

print("Step 7: Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("✅ Model training complete")

print("Step 8: Evaluating...")
y_pred = model.predict(X_test)
print("✅ Prediction complete")
print(classification_report(y_test, y_pred))

print("Step 9: Saving model...")
with open("upi_fraud_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("✅ Model saved as upi_fraud_model.pkl")
