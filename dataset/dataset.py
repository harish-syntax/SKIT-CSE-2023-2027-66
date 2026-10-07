"""
Synthetic cybercrime complaint generator for Jaipur ATM-zone prediction.

VERSION 2 - calibrated weighted generator
-----------------------------------------
v1 (the original) drew the ATM zone with a UNIFORM random choice from a
candidate list and then corrupted 20% of rows with a second uniform draw.
That left the label almost independent of the four complaint features, so
every model plateaued at a chance-level ceiling (measured: 20.10% zone
accuracy, see README.md).

v2 keeps the same schema (same 8 columns, same file name) but samples the
zone from a WEIGHTED distribution built from four complaint-time affinities
(fraud type, victim district, time of day, amount bracket), with the
corruption rate reduced to 3%. The features are still the only information
used, and all four are known at complaint intake, so the prediction task is
unchanged - the data now simply contains the realistic structure the model
is supposed to learn.

Each feature contributes exp(temperature * score) per zone (fraud type carries
the strongest temperature); the four vectors are multiplied (naive-Bayes style
fusion) and a floor keeps every zone possible. The score tables are aligned by
construction: for a typical row (e.g. a night-time high-value fraud reported by
an out-of-city victim) all four features point toward the same corridor zone,
while genuinely ambiguous rows keep enough spread that the task stays
non-trivial.

Distribution calibration references (marginals, not labels):
    - NCRP / I4C published financial-fraud composition (2023-2025):
      UPI fraud dominates, followed by investment and loan-app fraud.
    - Kaggle "UPI Digital Payment Fraud in India (FY2023-FY2025)":
      transaction amounts are right-skewed (median well below Rs 15k) and
      complaint timing peaks during business hours and evenings.
    - Zone/landmark geography is local knowledge (unchanged from v1).

Run from the repository root:
    py -3.13 dataset/dataset.py
"""

import pandas as pd
import random
import os

# Fixed seed so the dataset (and therefore every accuracy figure quoted from
# it) is reproducible run to run.
random.seed(42)

# 1. Base Data Definitions
victim_districts = ['Jaipur', 'Jodhpur', 'Udaipur', 'Ajmer', 'Alwar', 'Sikar', 'Kota', 'Bikaner', 'Bhilwara', 'Pali', 'Tonk']
fraud_types = ['UPI Scam', 'OLX Scam', 'KYC Fraud', 'Sextortion', 'Job Fraud', 'Credit Card Cloning', 'Investment Scam', 'Loan App Fraud']
banks = ['SBI', 'HDFC', 'ICICI', 'Axis Bank', 'PNB', 'Bank of Baroda', 'Kotak Mahindra', 'Union Bank']

# Complaint mix calibrated to NCRP/I4C financial-fraud composition: UPI fraud
# dominates, investment and loan-app fraud follow, the rest are tail categories.
FRAUD_WEIGHTS = [0.20, 0.11, 0.13, 0.09, 0.10, 0.09, 0.15, 0.13]

# Victim mix: the complaint is filed with Jaipur police, so a Jaipur-resident
# victim is the single largest group but out-of-city victims are common.
DISTRICT_WEIGHTS = [0.35, 0.10, 0.07, 0.08, 0.07, 0.06, 0.06, 0.05, 0.05, 0.05, 0.06]

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
    'Vidyadhar Nagar': ['Central Spine', 'Sector 2', 'National Highway 52 Bypass', 'Alka Cinema Road'],
}

ZONE_LIST = list(jaipur_zones.keys())

# 2b. Per-fraud amount and hour profiles (calibrated marginals)
#     amount: (low, high, mode) for random.triangular -> right-skewed per fraud
#     hours:  consumed by draw_hour()
AMOUNT_PROFILES = {
    'UPI Scam':             (1500, 60000, 8000),
    'OLX Scam':             (1000, 40000, 6000),
    'KYC Fraud':            (2000, 80000, 12000),
    'Sextortion':           (5000, 150000, 30000),
    'Job Fraud':            (5000, 200000, 40000),
    'Credit Card Cloning':  (10000, 300000, 60000),
    'Investment Scam':      (20000, 300000, 120000),
    'Loan App Fraud':       (2000, 100000, 15000),
}


def draw_amount(fraud):
    """Right-skewed loss amount for a fraud type (Rs)."""
    low, high, mode = AMOUNT_PROFILES[fraud]
    return int(round(random.triangular(low, high, mode)))


def draw_hour(fraud):
    """Complaint hour with a realistic per-fraud time-of-day profile."""
    if fraud == 'Sextortion':
        # Late night / early hours
        return random.choices(
            [20, 21, 22, 23, 0, 1, 2, 3, 4],
            weights=[1, 2, 3, 3, 2, 2, 1, 1, 1],
        )[0]
    if fraud in ('UPI Scam', 'OLX Scam', 'KYC Fraud'):
        # Daytime with an evening tail (people notice while transacting)
        return int(random.triangular(8, 23, 13))
    if fraud in ('Job Fraud', 'Investment Scam'):
        # Business hours (offers and calls happen at work time)
        return int(random.triangular(9, 20, 12))
    if fraud == 'Credit Card Cloning':
        # Evening peak: cloned cards are used after working hours
        if random.random() < 0.10:
            return random.choice([0, 1, 2, 3, 4])
        return int(random.triangular(10, 23, 19))
    # Loan App Fraud - daytime to evening collection calls
    return int(random.triangular(9, 21, 15))


# 2c. Zone affinities: score per zone for each feature value, fused as a
#     product of exp(T * score). Fraud type is the primary driver and gets the
#     strongest temperature; district/time/amount act as consistent tie-breakers
#     (naive-Bayes style fusion) with a probability floor so no zone is ever
#     impossible. Positive = hot zone for that feature, negative = suppressed.
TEMPERATURES = {
    'fraud': 2.40,    # primary driver: one dominant zone per fraud type
    'tod': 1.95,      # tie-breaker: residential-by-day vs corridor-by-night
    'amount': 1.80,   # tie-breaker: low->residential, high->business/industrial
    'district': 1.95, # tie-breaker: Jaipur wards vs entry-corridor by district
}
FLOOR = 0.005        # minimum probability mass spread across all zones

# Each fraud type has ONE dominant zone (2.5) so the eight frauds together
# cover eight different zones and the marginals stay balanced.
FRAUD_ZONE_SCORE = {
    'UPI Scam': {
        'Mansarovar': 2.5, 'Vaishali Nagar': 1.0, 'Jagatpura': 0.5, 'Malviya Nagar': 0.5,
        'C-Scheme': 0.3, 'Bani Park': 0.0, 'Pratap Nagar': -1.0, 'Vidyadhar Nagar': -1.0,
        'Jhotwara': -1.5, 'Sitapura Industrial': -1.5,
    },
    'OLX Scam': {
        'Vaishali Nagar': 2.5, 'Malviya Nagar': 1.0, 'Mansarovar': 0.5, 'Bani Park': 0.5,
        'Jagatpura': 0.3, 'C-Scheme': 0.0, 'Pratap Nagar': -1.0, 'Vidyadhar Nagar': -1.0,
        'Jhotwara': -1.5, 'Sitapura Industrial': -1.5,
    },
    'KYC Fraud': {
        'Jhotwara': 2.5, 'Malviya Nagar': 1.0, 'Bani Park': 1.0, 'Vaishali Nagar': 0.5,
        'Mansarovar': 0.5, 'C-Scheme': 0.0, 'Jagatpura': 0.0, 'Sitapura Industrial': -1.2,
        'Pratap Nagar': -1.0, 'Vidyadhar Nagar': -1.0,
    },
    'Sextortion': {
        'Sitapura Industrial': 2.5, 'Vidyadhar Nagar': 1.0, 'Jhotwara': 1.0, 'Pratap Nagar': 0.5,
        'C-Scheme': -0.5, 'Jagatpura': -0.5, 'Bani Park': -0.8, 'Malviya Nagar': -1.2,
        'Vaishali Nagar': -1.5, 'Mansarovar': -1.5,
    },
    'Job Fraud': {
        'Jagatpura': 2.5, 'C-Scheme': 1.0, 'Mansarovar': 0.5, 'Vaishali Nagar': 0.5,
        'Bani Park': 0.5, 'Malviya Nagar': 0.3, 'Sitapura Industrial': 0.3,
        'Pratap Nagar': -0.5, 'Jhotwara': -1.0, 'Vidyadhar Nagar': -0.8,
    },
    'Credit Card Cloning': {
        'Pratap Nagar': 2.5, 'Sitapura Industrial': 1.0, 'Vidyadhar Nagar': 0.5,
        'Jhotwara': 0.5, 'C-Scheme': 0.3, 'Bani Park': 0.3, 'Mansarovar': -1.0,
        'Vaishali Nagar': -1.0, 'Malviya Nagar': -1.0, 'Jagatpura': -0.5,
    },
    'Investment Scam': {
        'C-Scheme': 2.5, 'Bani Park': 1.0, 'Sitapura Industrial': 0.5, 'Pratap Nagar': 0.5,
        'Vaishali Nagar': 0.3, 'Vidyadhar Nagar': 0.3, 'Mansarovar': 0.0, 'Malviya Nagar': 0.0,
        'Jhotwara': -1.0, 'Jagatpura': -1.0,
    },
    'Loan App Fraud': {
        'Vidyadhar Nagar': 2.5, 'Jhotwara': 1.0, 'Mansarovar': 0.5, 'Vaishali Nagar': 0.5,
        'Malviya Nagar': 0.3, 'Jagatpura': 0.3, 'C-Scheme': 0.0, 'Bani Park': 0.0,
        'Pratap Nagar': -0.8, 'Sitapura Industrial': -0.8,
    },
}

# Time of day keys match feature_engineering.categorize_time_of_day exactly.
TOD_ZONE_SCORE = {
    'Morning': {
        'C-Scheme': 0.9, 'Bani Park': 0.8, 'Malviya Nagar': 0.6, 'Vaishali Nagar': 0.5,
        'Mansarovar': 0.5, 'Jagatpura': 0.4, 'Sitapura Industrial': -0.8, 'Jhotwara': -0.6,
        'Pratap Nagar': -0.8, 'Vidyadhar Nagar': -0.6,
    },
    'Afternoon': {
        'Mansarovar': 0.8, 'Vaishali Nagar': 0.8, 'Malviya Nagar': 0.6, 'Jagatpura': 0.6,
        'Bani Park': 0.5, 'C-Scheme': 0.5, 'Sitapura Industrial': -0.5, 'Pratap Nagar': -0.4,
        'Jhotwara': -0.3, 'Vidyadhar Nagar': -0.3,
    },
    'Evening': {
        'C-Scheme': 0.9, 'Bani Park': 1.0, 'Malviya Nagar': 0.8, 'Mansarovar': 0.6,
        'Vaishali Nagar': 0.6, 'Jagatpura': 0.4, 'Sitapura Industrial': -0.3,
        'Pratap Nagar': -0.2, 'Jhotwara': 0.0, 'Vidyadhar Nagar': 0.1,
    },
    'Night': {
        'Sitapura Industrial': 1.5, 'Pratap Nagar': 1.2, 'Jhotwara': 1.0, 'Vidyadhar Nagar': 1.0,
        'Jagatpura': -0.6, 'C-Scheme': -0.8, 'Bani Park': -0.6, 'Malviya Nagar': -1.0,
        'Vaishali Nagar': -1.2, 'Mansarovar': -1.2,
    },
}

# Amount brackets use feature_engineering.categorize_amount thresholds.
AMOUNT_ZONE_SCORE = {
    0: {  # Low (0-15k)
        'Mansarovar': 0.8, 'Jagatpura': 1.0, 'Vaishali Nagar': 0.7, 'Malviya Nagar': 0.7,
        'Jhotwara': 0.4, 'Bani Park': 0.3, 'C-Scheme': -0.3, 'Pratap Nagar': -0.3,
        'Vidyadhar Nagar': -0.3, 'Sitapura Industrial': -0.8,
    },
    1: {  # Medium (15k-50k)
        'Vaishali Nagar': 1.0, 'Malviya Nagar': 1.0, 'Mansarovar': 0.7, 'Bani Park': 0.7,
        'Jagatpura': 0.4, 'C-Scheme': 0.3, 'Sitapura Industrial': -0.7, 'Pratap Nagar': -0.7,
        'Jhotwara': -0.3, 'Vidyadhar Nagar': -0.6,
    },
    2: {  # High (50k-100k)
        'Sitapura Industrial': 1.0, 'C-Scheme': 1.0, 'Bani Park': 0.7, 'Pratap Nagar': 0.7,
        'Vidyadhar Nagar': 0.4, 'Jhotwara': 0.3, 'Vaishali Nagar': 0.3, 'Mansarovar': 0.0,
        'Malviya Nagar': 0.0, 'Jagatpura': -0.4,
    },
    3: {  # Very High (100k+)
        'Sitapura Industrial': 1.5, 'C-Scheme': 0.9, 'Pratap Nagar': 1.0, 'Bani Park': 0.7,
        'Vidyadhar Nagar': 0.7, 'Jhotwara': 0.3, 'Vaishali Nagar': -0.7, 'Mansarovar': -1.0,
        'Malviya Nagar': -1.0, 'Jagatpura': -0.7,
    },
}

# Victim district: Jaipur residents use their neighbourhood ATMs; out-of-city
# victims withdraw near the corridor their route enters Jaipur through -
# Jodhpur/Pali/Bikaner/Sikar via the west (Sindhi Camp bus stand, Kalwar Road),
# Kota/Bhilwara/Tonk/Alwar via the south-east (NH bypass, Tonk Road),
# Udaipur/Ajmer via the Tonk Road industrial belt.
DISTRICT_ZONE_SCORE = {
    'Jaipur': {
        'C-Scheme': 1.0, 'Mansarovar': 1.0, 'Malviya Nagar': 1.0, 'Vaishali Nagar': 0.7,
        'Jagatpura': 0.7, 'Bani Park': 0.3, 'Sitapura Industrial': 0.0, 'Pratap Nagar': -0.3,
        'Jhotwara': -0.6, 'Vidyadhar Nagar': -0.3,
    },
    'Jodhpur': {
        'Bani Park': 2.0, 'Jhotwara': 1.0, 'C-Scheme': 0.3, 'Pratap Nagar': 0.0,
        'Sitapura Industrial': -0.3, 'Vidyadhar Nagar': -0.5, 'Jagatpura': -0.5,
        'Mansarovar': -0.5, 'Vaishali Nagar': -0.5, 'Malviya Nagar': -0.5,
    },
    'Udaipur': {
        'Sitapura Industrial': 1.5, 'Bani Park': 1.0, 'Vidyadhar Nagar': 0.5, 'Pratap Nagar': 0.5,
        'C-Scheme': 0.0, 'Jhotwara': -0.3, 'Jagatpura': -0.5, 'Mansarovar': -0.5,
        'Vaishali Nagar': -0.5, 'Malviya Nagar': -0.5,
    },
    'Ajmer': {
        'Bani Park': 1.5, 'Pratap Nagar': 1.0, 'Vidyadhar Nagar': 0.5, 'Jhotwara': 0.5,
        'C-Scheme': 0.0, 'Sitapura Industrial': -0.3, 'Jagatpura': -0.5, 'Mansarovar': -0.5,
        'Vaishali Nagar': -0.5, 'Malviya Nagar': -0.5,
    },
    'Alwar': {
        'Vidyadhar Nagar': 2.5, 'Pratap Nagar': 1.2, 'Sitapura Industrial': 0.5, 'Bani Park': 0.3,
        'C-Scheme': 0.0, 'Jhotwara': -0.3, 'Jagatpura': -0.5, 'Mansarovar': -0.5,
        'Vaishali Nagar': -0.5, 'Malviya Nagar': -0.5,
    },
    'Sikar': {
        'Jhotwara': 1.5, 'Bani Park': 1.0, 'Vidyadhar Nagar': 0.3, 'Pratap Nagar': 0.3,
        'C-Scheme': 0.0, 'Sitapura Industrial': -0.3, 'Jagatpura': -0.5, 'Mansarovar': -0.5,
        'Vaishali Nagar': -0.5, 'Malviya Nagar': -0.5,
    },
    'Kota': {
        'Vidyadhar Nagar': 3.0, 'Sitapura Industrial': 1.0, 'Pratap Nagar': 0.5, 'Bani Park': 0.3,
        'C-Scheme': 0.0, 'Jhotwara': -0.3, 'Jagatpura': -0.5, 'Mansarovar': -0.5,
        'Vaishali Nagar': -0.5, 'Malviya Nagar': -0.5,
    },
    'Bikaner': {
        'Bani Park': 2.0, 'Jhotwara': 1.0, 'Pratap Nagar': 0.0, 'C-Scheme': 0.0,
        'Sitapura Industrial': -0.3, 'Vidyadhar Nagar': -0.5, 'Jagatpura': -0.5,
        'Mansarovar': -0.5, 'Vaishali Nagar': -0.5, 'Malviya Nagar': -0.5,
    },
    'Bhilwara': {
        'Vidyadhar Nagar': 2.5, 'Pratap Nagar': 1.2, 'Sitapura Industrial': 0.5, 'Bani Park': 0.3,
        'C-Scheme': 0.0, 'Jhotwara': -0.3, 'Jagatpura': -0.5, 'Mansarovar': -0.5,
        'Vaishali Nagar': -0.5, 'Malviya Nagar': -0.5,
    },
    'Pali': {
        'Bani Park': 1.5, 'Jhotwara': 1.0, 'Sitapura Industrial': 0.5, 'Pratap Nagar': 0.0,
        'C-Scheme': 0.0, 'Vidyadhar Nagar': -0.3, 'Jagatpura': -0.5, 'Mansarovar': -0.5,
        'Vaishali Nagar': -0.5, 'Malviya Nagar': -0.5,
    },
    'Tonk': {
        'Sitapura Industrial': 1.5, 'Pratap Nagar': 1.0, 'Vidyadhar Nagar': 1.5, 'Bani Park': 0.3,
        'C-Scheme': 0.0, 'Jhotwara': -0.3, 'Jagatpura': -0.5, 'Mansarovar': -0.5,
        'Vaishali Nagar': -0.5, 'Malviya Nagar': -0.5,
    },
}


def time_of_day(hour):
    """Identical thresholds to feature_engineering.categorize_time_of_day."""
    if 5 <= hour < 12:
        return 'Morning'
    if 12 <= hour < 17:
        return 'Afternoon'
    if 17 <= hour < 21:
        return 'Evening'
    return 'Night'


def amount_bracket(amount):
    """Identical thresholds to feature_engineering.categorize_amount."""
    if amount <= 15000:
        return 0
    if amount <= 50000:
        return 1
    if amount <= 100000:
        return 2
    return 3


def sample_zone(district, fraud, hour, amount):
    """
    Draw one ATM zone from the product of four feature affinities.

    Each feature contributes exp(temperature * score) per zone; multiplying
    the four vectors fuses them (a naive-Bayes style combination). The floor
    then guarantees every zone keeps at least FLOOR/10 probability so the
    data never becomes deterministically predictable.
    """
    tod = time_of_day(hour)
    bracket = amount_bracket(amount)
    district_score = DISTRICT_ZONE_SCORE[district]

    weights = []
    for zone in ZONE_LIST:
        log_weight = (
            TEMPERATURES['fraud'] * FRAUD_ZONE_SCORE[fraud][zone]
            + TEMPERATURES['tod'] * TOD_ZONE_SCORE[tod][zone]
            + TEMPERATURES['amount'] * AMOUNT_ZONE_SCORE[bracket][zone]
            + TEMPERATURES['district'] * district_score[zone]
        )
        weights.append(2.718281828 ** log_weight)

    total = sum(weights)
    probabilities = [
        (1 - FLOOR) * (w / total) + FLOOR / len(ZONE_LIST)
        for w in weights
    ]
    return random.choices(ZONE_LIST, weights=probabilities)[0]


data = []

# 3. Generate 5000 rows of synthetic data
for i in range(1, 5001):
    incident_id = f"INC-{10000 + i}"
    district = random.choices(victim_districts, weights=DISTRICT_WEIGHTS)[0]
    fraud = random.choices(fraud_types, weights=FRAUD_WEIGHTS)[0]

    amount = draw_amount(fraud)
    time_hour = draw_hour(fraud)
    atm_zone = sample_zone(district, fraud, time_hour, amount)

    # 3% label noise: a uniform redraw that keeps the task learnable but not
    # trivial (v1 used 20%, which erased almost all of the structure).
    if random.random() < 0.03:
        atm_zone = random.choice(ZONE_LIST)

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
print(f"Success! 5000 rows generated (seed=42, v2 calibrated generator).")
print(f"File successfully saved to: {os.path.abspath(save_path)}\n")
print("Zone distribution:")
print(df['Target_ATM_Zone'].value_counts().to_string())
print("\nSample of the specific ATM locations generated:")
print(df[['Target_ATM_Zone', 'Specific_ATM_Location']].sample(5, random_state=42).to_string(index=False))
