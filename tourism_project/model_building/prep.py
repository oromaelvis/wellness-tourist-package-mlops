import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"
TARGET = "ProdTaken"

df = pd.read_csv(DATA_PATH)
print(f"Loaded {DATA_PATH}: {df.shape}")

# remove columns that are not useful predictors
#    - Unnamed: 0 and CustomerID are identifiers
#    - Designation maps 1:1 onto ProductPitched, so it adds no new information
cols_to_drop = [c for c in ["Unnamed: 0", "CustomerID", "Designation"] if c in df.columns]
df = df.drop(columns=cols_to_drop)
print("Dropped columns:", cols_to_drop)

# fix inconsistent category labels
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
df["Occupation"] = df["Occupation"].replace({"Free Lancer": "Freelancer"})

# remove duplicates and incomplete rows
before = len(df)
df = df.drop_duplicates()
df = df.dropna()
print(f"Removed {before - len(df)} duplicate/incomplete rows. Remaining: {len(df)}")

# stratified split so both sets keep the same ~19% buyer ratio
X = df.drop(columns=[TARGET])
y = df[TARGET]
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# save the splits, the workflow passes these to the training job as an artifact
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print(f"Train: {Xtrain.shape}, Test: {Xtest.shape}")
print(f"Buyer ratio - train: {ytrain.mean():.3f}, test: {ytest.mean():.3f}")
print("Data prepared: Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv written.")
