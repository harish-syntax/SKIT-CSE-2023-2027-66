import pandas as pd
import os

# Ensure the dataset folder exists just in case
os.makedirs('dataset', exist_ok=True)

# 1. Load the original dataset from the dataset folder
# (Make sure this filename matches exactly what you generated earlier)
file_path = 'dataset/jaipur_cybercrime_5000_detailed.csv'
df = pd.read_csv(file_path)

print("Starting Manual Feature Engineering...\n")

# =======================================================
# STEP 1: Manually Extract Hour and Time of Day
# =======================================================
def extract_hour(time_string):
    # Split "14:30" by the colon and take the first part "14"
    hour_part = time_string.split(':')[0]
    return int(hour_part)

def categorize_time_of_day(hour):
    if hour >= 5 and hour < 12:
        return 'Morning'
    elif hour >= 12 and hour < 17:
        return 'Afternoon'
    elif hour >= 17 and hour < 21:
        return 'Evening'
    else:
        return 'Night'

# Apply our custom functions to the DataFrame
df['Hour'] = df['Time_of_Complaint'].apply(extract_hour)
df['Time_of_Day'] = df['Hour'].apply(categorize_time_of_day)


# =======================================================
# STEP 2: Manually Create Amount Brackets
# =======================================================
def categorize_amount(amount):
    if amount <= 15000:
        return 'Low (0-15k)'
    elif amount > 15000 and amount <= 50000:
        return 'Medium (15k-50k)'
    elif amount > 50000 and amount <= 100000:
        return 'High (50k-100k)'
    else:
        return 'Very High (100k+)'

df['Amount_Bracket'] = df['Amount_INR'].apply(categorize_amount)


# =======================================================
# STEP 3: Manually Convert Text to Numbers (Encoding)
# =======================================================
# A. Dictionary for Fraud Types
fraud_mapping = {
    'UPI Scam': 0, 
    'OLX Scam': 1, 
    'KYC Fraud': 2, 
    'Sextortion': 3,
    'Job Fraud': 4, 
    'Credit Card Cloning': 5, 
    'Investment Scam': 6, 
    'Loan App Fraud': 7
}

# B. Dictionary for Victim Districts
district_mapping = {
    'Jaipur': 0, 'Jodhpur': 1, 'Udaipur': 2, 'Ajmer': 3, 
    'Alwar': 4, 'Sikar': 5, 'Kota': 6, 'Bikaner': 7, 
    'Bhilwara': 8, 'Pali': 9, 'Tonk': 10
}

# C. Dictionary for Time of Day
time_mapping = {
    'Morning': 0, 
    'Afternoon': 1, 
    'Evening': 2, 
    'Night': 3
}

# D. Dictionary for Amount Brackets
amount_mapping = {
    'Low (0-15k)': 0, 
    'Medium (15k-50k)': 1, 
    'High (50k-100k)': 2, 
    'Very High (100k+)': 3
}

# E. Dictionary for Target ATM Zones (Our prediction labels)
zone_mapping = {
    'Mansarovar': 0, 
    'Sitapura Industrial': 1, 
    'Vaishali Nagar': 2,
    'Malviya Nagar': 3, 
    'Jhotwara': 4, 
    'C-Scheme': 5, 
    'Pratap Nagar': 6,
    'Bani Park': 7, 
    'Jagatpura': 8, 
    'Vidyadhar Nagar': 9
}

# Now, map these dictionaries to create new numerical columns
df['Fraud_Type_Code'] = df['Fraud_Type'].map(fraud_mapping)
df['Victim_District_Code'] = df['Victim_District'].map(district_mapping)
df['Time_of_Day_Code'] = df['Time_of_Day'].map(time_mapping)
df['Amount_Bracket_Code'] = df['Amount_Bracket'].map(amount_mapping)
df['Target_ATM_Zone_Code'] = df['Target_ATM_Zone'].map(zone_mapping)


# =======================================================
# STEP 4: Select Final Columns and Save
# =======================================================
# Keep only the numbered columns that the ML model needs to train
# Added 'Incident_ID' just so you can track which row is which, but you can remove it if you prefer.
final_columns_for_ml = [
    'Incident_ID',
    'Fraud_Type_Code', 
    'Victim_District_Code', 
    'Time_of_Day_Code', 
    'Amount_Bracket_Code', 
    'Target_ATM_Zone_Code'
]

df_ml_ready = df[final_columns_for_ml]

# Save the explicitly engineered dataset as a new file in the dataset folder
output_path = 'dataset/jaipur_ml_ready_manual_features.csv'
df_ml_ready.to_csv(output_path, index=False)

print("Manual Feature Engineering Complete!")
print(f"Data ready for Machine Learning saved to: {output_path}\n")

# Show the first 5 rows to verify the numbers
print("Preview of the numerical data (Ready for Training):")
print(df_ml_ready.head())