# -------------------------------------------------------
# HOSTING SCRIPT
# Pushes all deployment files to the Hugging Face Space, which
# rebuilds the Docker container and serves the Streamlit app.
# ------------------------------------------------------
from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
# Connecting to Hugging Face. The token comes from the environment variable

api.upload_folder(
    folder_path="tourism_project/deployment",              # local folder with the app files
    repo_id="ivkobenko/tourism-package-prediction",    # the target HF Space
    repo_type="space",                                     # repo type: space
    path_in_repo="",                                       # upload to the root of the Space
)
print("Deployment files pushed to the Hugging Face Space.")
