import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Authenticate using the Kaggle API
api = KaggleApi()
api.authenticate()

# Download the dataset
dataset_owner = 'snehaanbhawal'
dataset_name = 'resume-dataset'

# The directory to store files in
output_dir = "./llamaindex-resumes/"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Download the dataset
api.dataset_download_files(f'{dataset_owner}/{dataset_name}', path=output_dir, unzip=True)

print("Dataset downloaded successfully!")
