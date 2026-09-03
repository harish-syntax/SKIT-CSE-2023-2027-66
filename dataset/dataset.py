import pandas as pd
import random
import os

# 1. Base Data Definitions
victim_districts = ['Jaipur', 'Jodhpur', 'Udaipur', 'Ajmer', 'Alwar', 'Sikar', 'Kota', 'Bikaner', 'Bhilwara', 'Pali', 'Tonk']
fraud_types = ['UPI Scam', 'OLX Scam', 'KYC Fraud', 'Sextortion', 'Job Fraud', 'Credit Card Cloning', 'Investment Scam', 'Loan App Fraud']
banks = ['SBI', 'HDFC', 'ICICI', 'Axis Bank', 'PNB', 'Bank of Baroda', 'Kotak Mahindra', 'Union Bank']

# 2. Jaipur Zones and their Specific Real-World Landmarks/Streets
jaipur_zones = {
    'Mansarovar': ['Madhyam Marg', 'VT Road Chauraha', 'Patrakar Colony', 'Kissan Dharam Kanta', 'Kaveri Path', 'Rajat Path'],
    'Sitapura Industrial': ['RIICO Phase 1', 'India Gate', 'EPI Zone', 'Tonk Road Junction', 'Mahatma Gandhi Hospital Road'],
    'Vaishali Nagar': ['Amrapali Circle', 'Gandhi Path', 'Queens Road', 'Nursery Circle', 'Hanuman Nagar'],
    'Malviya Nagar': ['Gaurav Tower (GT) Road', 'Calgiri Road', 'Apex Circle', 'JLN Marg', 'Sector 3 Market'],
    'Jhotwara': ['Kalwar Road', 'Panchawala', 'Kanta Chauraha', 'Triton Mall Road', 'Lata Circle'],
    'C-Scheme': ['Ahinsa Circle', 'Statue Circle', 'Ashok Marg', 'Subhash Marg', 'MI Road Junction'],
    'Pratap Nagar': ['Haldi Ghati Marg', 'Kumbha Marg', 'Sector 11', 'Sector 16 Market', 'Tonk Road'],
    'Bani Park': ['Collectorate Circle', 'Sindhi Camp Bus Stand Road', 'Kabir Marg', 'Peetal Factory'],
    'Jagatpura': ['Mahal Road', 'NRI Colony', 'SKIT College Road', 'Ramnagariya', '7 Number Stand'],
    'Vidyadhar Nagar': ['Central Spine', 'Sector 2', 'National Highway 52 Bypass', 'Alka Cinema Road']
}

data = []

# 3. Generate 5000 rows of synthetic data
for i in range(1, 5001):
    incident_id = f"INC-{10000 + i}"
    district = random.choice(victim_districts)
    fraud = random.choice(fraud_types)
    
    # ML LOGIC: Establish realistic criminal patterns
    if fraud in ['Sextortion', 'Credit Card Cloning', 'Investment Scam']:
        # High value, high risk -> Mules prefer isolated, industrial, or highway-adjacent areas, mostly at night
        amount = random.randint(50000, 300000)
        atm_zone = random.choice(['Sitapura Industrial', 'Pratap Nagar', 'Jhotwara', 'Vidyadhar Nagar']) 
        time_hour = random.choice([22, 23, 0, 1, 2, 3, 4]) # Night time
    elif fraud in ['OLX Scam', 'UPI Scam', 'KYC Fraud']:
        # Lower value, quick cashouts -> Dense residential areas, during the day to blend in with crowds
        amount = random.randint(2000, 30000)
        atm_zone = random.choice(['Mansarovar', 'Vaishali Nagar', 'Bani Park', 'Malviya Nagar'])
        time_hour = random.randint(9, 20) # Day/Evening time
    else:
        # General mixed frauds
        amount = random.randint(15000, 100000)
        atm_zone = random.choice(list(jaipur_zones.keys()))
        time_hour = random.randint(0, 23)
        
    # Introduce 20% randomness (noise) so the ML model has to work hard and isn't 100% perfect
    if random.random() < 0.20:
        atm_zone = random.choice(list(jaipur_zones.keys()))
        time_hour = random.randint(0, 23)

    # 4. Generate Specific ATM Location
    bank = random.choice(banks)
    landmark = random.choice(jaipur_zones[atm_zone])
    specific_atm = f"{bank} ATM, {landmark}, {atm_zone}"

    # Generate random timestamp
    minute = random.randint(0, 59)
    time_str = f"{time_hour:02d}:{minute:02d}"
    
    # Time to withdraw based on amount and fraud type
    time_to_withdraw = "1-2 Hours" if amount < 30000 else "12-24 Hours"

    data.append([incident_id, district, fraud, amount, time_str, time_to_withdraw, atm_zone, specific_atm])

# 5. Create DataFrame
columns = ['Incident_ID', 'Victim_District', 'Fraud_Type', 'Amount_INR', 'Time_of_Complaint', 'Time_to_Withdraw', 'Target_ATM_Zone', 'Specific_ATM_Location']
df = pd.DataFrame(data, columns=columns)

# 6. Folder Creation and Saving Logic
output_folder = "dataset"
output_file = "jaipur_cybercrime_5000_detailed.csv"

# Automatically create the 'dataset' folder if it doesn't already exist
os.makedirs(output_folder, exist_ok=True)

# Combine folder and file name into a full path
save_path = os.path.join(output_folder, output_file)

# Save the DataFrame
df.to_csv(save_path, index=False)

# Print confirmation and sample
print(f"Success! 5000 rows generated.")
print(f"File successfully saved to: {os.path.abspath(save_path)}\n")
print("Sample of the specific ATM locations generated:")
print(df[['Target_ATM_Zone', 'Specific_ATM_Location']].sample(5).to_string(index=False))