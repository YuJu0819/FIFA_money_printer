import os
import shutil

import kagglehub

# Download latest version (goes into kagglehub's cache and returns that path)
path = kagglehub.dataset_download(
    "martj42/international-football-results-from-1872-to-2017")

print("Path to dataset files:", path)

# Copy the downloaded files into the current working folder
dest = os.path.join(os.getcwd(), "data")
shutil.copytree(path, dest, dirs_exist_ok=True)

print("Dataset copied to:", dest)
