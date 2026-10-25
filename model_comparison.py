import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from xgboost import XGBClassifier

from atm_zone_config import (
    INPUT_FILE,
    FEATURES,
    TARGET,
    RANDOM_STATE,
    XGB_PARAMS,
)


def top_3_hit_rate(model, X_test, y_test):
    probabilities = model.predict_proba(X_test)
    top_3_indices = np.argsort(probabilities, axis=1)[:, -3:]

    actual_labels = np.asarray(y_test)
    hits = [
        actual_labels[i] in top_3_indices[i]
        for i in range(len(actual_labels))
    ]

    return np.mean(hits)


def evaluate_model(name, model, X_test, y_test):
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )
    top_3 = top_3_hit_rate(model, X_test, y_test)

    print(f"\n{name}")
    print("-" * 45)
    print(f"Exact-zone accuracy : {accuracy:.4f} ({accuracy:.2%})")
    print(f"Macro-F1             : {macro_f1:.4f}")
    print(f"Top-3 hit rate       : {top_3:.4f} ({top_3:.2%})")

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Macro-F1": macro_f1,
        "Top-3 Hit Rate": top_3,
    }


def main():
    df = pd.read_csv(INPUT_FILE)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("=" * 55)
    print("FAIR RANDOM FOREST VS XGBOOST COMPARISON")
    print("=" * 55)
    print(f"Dataset       : {INPUT_FILE}")
    print(f"Total rows    : {len(df)}")
    print(f"Training rows : {len(X_train)}")
    print(f"Testing rows  : {len(X_test)}")
    print(f"Features      : {', '.join(FEATURES)}")
    print(f"Random seed   : {RANDOM_STATE}")

    # Best Random Forest settings from your previous tuning.
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=2,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )

    print("\nTraining tuned Random Forest...")
    rf_model.fit(X_train, y_train)

    # Plain XGBoost using the shared project configuration.
    xgb_model = XGBClassifier(
        **XGB_PARAMS,
        objective="multi:softprob",
        num_class=10,
        eval_metric="mlogloss",
    )

    print("Training plain XGBoost...")
    xgb_model.fit(X_train, y_train)

    results = [
        evaluate_model(
            "Tuned Random Forest", rf_model, X_test, y_test
        ),
        evaluate_model(
            "Plain XGBoost", xgb_model, X_test, y_test
        ),
    ]

    print("\n" + "=" * 55)
    print("FINAL COMPARISON")
    print("=" * 55)

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False, float_format="%.4f"))

    print(
        "\nNote: These are test results on the project's existing "
        "dataset. They do not establish real-world forecasting accuracy."
    )


if __name__ == "__main__":
    main()
