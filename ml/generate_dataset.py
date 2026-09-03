import os

import numpy as np
import pandas as pd


# Make the generated dataset reproducible
np.random.seed(42)

# Number of synthetic complaints
NUM_RECORDS = 1000


# Possible cybercrime locations
zones = [
    "Zone_A",
    "Zone_B",
    "Zone_C",
    "Zone_D",
    "Zone_E",
    "Zone_F",
    "Zone_G",
    "Zone_H"
]


# Types of cybercrime complaints
complaint_types = [
    "UPI_Fraud",
    "Card_Fraud",
    "Phishing",
    "Online_Shopping_Fraud",
    "Investment_Fraud"
]


# Payment methods involved in complaints
payment_methods = [
    "UPI",
    "Debit_Card",
    "Credit_Card",
    "Net_Banking",
    "Wallet"
]


# Generate random dates during 2025
dates = pd.to_datetime(
    np.random.choice(
        pd.date_range("2025-01-01", "2025-12-31"),
        NUM_RECORDS
    )
)


# Generate random complaint hours
hours = np.random.randint(0, 24, NUM_RECORDS)


# Create the synthetic dataset
data = pd.DataFrame({
    "complaint_id": range(1, NUM_RECORDS + 1),

    "date": dates,

    "location_zone": np.random.choice(
        zones,
        NUM_RECORDS
    ),

    "complaint_type": np.random.choice(
        complaint_types,
        NUM_RECORDS
    ),

    "payment_method": np.random.choice(
        payment_methods,
        NUM_RECORDS
    ),

    "transaction_amount": np.random.randint(
        500,
        100000,
        NUM_RECORDS
    ),

    "withdrawal_zone": np.random.choice(
        zones,
        NUM_RECORDS
    ),

    "hour": hours
})


# Extract useful time-related features
data["day_of_week"] = data["date"].dt.dayofweek
data["month"] = data["date"].dt.month


# Create the data directory if it doesn't exist
os.makedirs("ml/data", exist_ok=True)


# Save the dataset as a CSV file
output_path = "ml/data/cybercrime_data.csv"

data.to_csv(
    output_path,
    index=False
)


print(f"Generated {len(data)} synthetic complaint records.")
print(f"Dataset saved to: {output_path}")
print("\nDataset columns:")
print(data.columns.tolist())

print("\nFirst 5 records:")
print(data.head())