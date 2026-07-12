# --------------------------------------
# DATA PREPARATION SCRIPT
# Loading raw data from the HF dataset space, cleans it, splits it
# into train/test, and uploads the splits back to the HF data space
# ------------------------------------
# for data manipulation
import pandas as pd
import sklearn

# for creating folders / reading env variables
import os

# for the train-test split
from sklearn.model_selection import train_test_split

# for Hugging Face authentication and file upload
from huggingface_hub import login, HfApi

# Initialize the API client (token from the environment variable HF_TOKEN)
api = HfApi(token=os.getenv("HF_TOKEN"))

# --------------------1. LOAD THE DATASET ------------
# Load the raw data DIRECTLY from the Hugging Face dataset space
DATASET_PATH = "hf://datasets/ivkobenko/tourism-package-prediction/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully. Shape:", df.shape)

# ----------------- 2. DATA CLEANING --------------
# Drop unnecessary columns:
#  - 'Unnamed: 0' is a leftover export index (pure row number, no signal)
#  - 'CustomerID' is a unique identifier (cannot generalise, causes overfitting)
df.drop(columns=["Unnamed: 0", "CustomerID"], inplace=True, errors="ignore")

# Fix the data-entry error in the Gender column: 'Fe Male' -> 'Female'
df["Gender"] = df["Gender"].replace("Fe Male", "Female")

# Imputation for future -  automated pipeline never fails on new data:
#  - numeric columns - median (robust to outliers)
#  - categorical columns - mode (most frequent value)
for col in df.columns:
    if df[col].isnull().any():
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
print("Cleaning complete. Remaining missing values:", int(df.isnull().sum().sum()))

# --------------- 3. TRAIN / TEST SPLIT -------------------
target_col = "ProdTaken"   # 1 = purchased the package, 0 = not purchased 

# Split into features (X) and target (y)
X = df.drop(columns=[target_col])
y = df[target_col]

# 80/20 split; stratify on y to preserve the ~19% buyer ratio in both sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# tratify=y forces both sets to keep the same 19%/81% buyer ratio — without it, 
# random chance could give the test set 15% or 24% buyers and distort the metrics.

print("Train shape:", Xtrain.shape, "| Test shape:", Xtest.shape)

# Save the four splits locally as CSV files
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# ------------- 4. UPLOAD SPLITS TO Hugging Face------------------
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],   # just the filename
        repo_id="ivkobenko/tourism-package-prediction",
        repo_type="dataset",
    )
print("Train/test splits uploaded back to the Hugging Face data space.")
