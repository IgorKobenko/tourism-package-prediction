# ---------------------------------------------------------------
# DATA REGISTRATION SCRIPT
# Registers the raw tourism dataset on the Hugging Face dataset hub
# ---------------------------------------------------------------
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

# Target dataset repository on Hugging Face: "<username>/<dataset-name>"
repo_id = "ivkobenko/tourism-package-prediction"
repo_type = "dataset"

# Initialize the API client; the token is read from the environment variable HF_TOKEN
# (set locally in Colab, or injected by GitHub Actions from the repo secret)
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the dataset repository already exists on the Hub
# First run creates the repo; every later run just reuses it.
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repo '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    # Step 2: If it does not exist, create a new public dataset repository
    print(f"Dataset repo '{repo_id}' not found. Creating a new one...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset repo '{repo_id}' created.")

# Step 3: Upload the entire local data folder (tourism.csv) to the dataset repo
api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
print("Dataset uploaded to the Hugging Face dataset space successfully.")
