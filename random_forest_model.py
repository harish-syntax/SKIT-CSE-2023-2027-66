import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


INPUT_FILE = "dataset/jaipur_ml_ready_manual_features.csv"


# Load ML-ready dataset
df = pd.read_csv(INPUT_FILE)

# Features used for prediction
features = [
    "Fraud_Type_Code",
    "Victim_District_Code",
    "Time_of_Day_Code",
    "Amount_Bracket_Code"
]

# Target: ATM withdrawal zone
target = "Target_ATM_Zone_Code"

X = df[features]
y = df[target]


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Create Random Forest baseline model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

# Train the model
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("Random Forest Baseline Model")
print("-" * 35)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Number of features: {len(features)}")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

print("\nFeature Importance:")
for feature, importance in zip(features, model.feature_importances_):
    print(f"{feature}: {importance:.4f}")

print("\nN_estimators Comparison:")

for n in [50, 100, 200, 300]:
    test_model = RandomForestClassifier(
        n_estimators=n,
        random_state=42,
        class_weight="balanced"
    )

    test_model.fit(X_train, y_train)
    test_pred = test_model.predict(X_test)

    test_accuracy = accuracy_score(y_test, test_pred)

    print(f"{n} trees -> Accuracy: {test_accuracy:.4f}")