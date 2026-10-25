import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

from atm_zone_config import (
    FEATURES,
    TARGET,
    RANDOM_STATE,
    XGB_PARAMS,
    INPUT_FILE,
)


def main():
    df = pd.read_csv(INPUT_FILE)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("XGBoost Model Evaluation")
    print("------------------------")
    print(f"Test samples: {len(y_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))


if __name__ == "__main__":
    main()