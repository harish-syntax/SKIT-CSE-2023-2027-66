import pandas as pd
from pathlib import Path


INPUT_FILE = Path("dataset/jaipur_cybercrime_5000_detailed.csv")
OUTPUT_FILE = Path("dataset/jaipur_cybercrime_cleaned.csv")


def clean_data(df):
    # Standardize column names
    df.columns = df.columns.str.strip()

    # Remove unwanted spaces from text columns
    text_columns = [
        "Incident_ID",
        "Victim_District",
        "Fraud_Type",
        "Time_to_Withdraw",
        "Target_ATM_Zone",
        "Specific_ATM_Location"
    ]

    for column in text_columns:
        df[column] = df[column].astype(str).str.strip()

    # Convert amount to numeric
    df["Amount_INR"] = pd.to_numeric(
        df["Amount_INR"], errors="coerce"
    )

    # Validate complaint time
    parsed_time = pd.to_datetime(
        df["Time_of_Complaint"],
        format="%H:%M",
        errors="coerce"
    )

    df["Time_of_Complaint"] = parsed_time.dt.strftime("%H:%M")

    # Remove duplicate incident IDs
    df = df.drop_duplicates(
        subset=["Incident_ID"],
        keep="first"
    )

    # Remove records with invalid required values
    required_columns = [
        "Incident_ID",
        "Victim_District",
        "Fraud_Type",
        "Amount_INR",
        "Time_of_Complaint",
        "Time_to_Withdraw",
        "Target_ATM_Zone"
    ]

    df = df.dropna(subset=required_columns)

    # Transaction amount must be positive
    df = df[df["Amount_INR"] > 0]

    return df


def main():
    print("Loading dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Original rows: {len(df)}")
    print(f"Original columns: {len(df.columns)}")
    print(f"Missing values before cleaning: {df.isnull().sum().sum()}")
    print(f"Duplicate rows before cleaning: {df.duplicated().sum()}")

    cleaned_df = clean_data(df)

    print(f"Rows after cleaning: {len(cleaned_df)}")
    print(f"Rows removed: {len(df) - len(cleaned_df)}")
    print(
        f"Missing values after cleaning: "
        f"{cleaned_df.isnull().sum().sum()}"
    )
    print(
        f"Duplicate rows after cleaning: "
        f"{cleaned_df.duplicated().sum()}"
    )

    cleaned_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nCleaned dataset saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()