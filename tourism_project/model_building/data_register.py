import sys
import pandas as pd

RAW_PATH = "tourism_project/data/tourism.csv"

# columns the rest of the pipeline relies on
EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]

df = pd.read_csv(RAW_PATH)

# schema check to make sure nothing is missing
missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
if missing:
    sys.exit(f"Registration failed. Missing columns: {missing}")

# target sanity check
if not set(df["ProdTaken"].unique()).issubset({0, 1}):
    sys.exit("Registration failed. ProdTaken should only contain 0 and 1.")

# registration summary
print("Dataset registered successfully.")
print(f"Source file : {RAW_PATH}")
print(f"Rows        : {df.shape[0]}")
print(f"Columns     : {df.shape[1]}")
extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
if extra:
    print(f"Extra columns (will be dropped in prep): {extra}")
print(f"Missing values in expected columns: {int(df[EXPECTED_COLUMNS].isna().sum().sum())}")
print(f"Duplicate CustomerIDs : {int(df['CustomerID'].duplicated().sum())}")
print("Target (ProdTaken) distribution:")
print(df["ProdTaken"].value_counts().rename(index={0: "0 - did not buy", 1: "1 - bought"}).to_string())
