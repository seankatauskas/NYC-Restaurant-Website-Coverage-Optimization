# make_ordinal() and update_address() is adapted from Bulk Geocoding with Geopy and GeoPanda(https://github.com/spatialthoughts/python-tutorials/geocoding_with_geopy.ipynb)
# Original code by Ujaval Gandhi, 2022
# Source: https://github.com/spatialthoughts/python-tutorials/blob/main/LICENSE
# Licensed under the MIT License (https://github.com/spatialthoughts/python-tutorials/blob/main/LICENSE)

import re
import os
import requests
import zipfile
import shutil
import pandas as pd

# Function converting street/avenue numbers to ordinals
def make_ordinal(match):
    n = int(match.group(1))
    if 11 <= (n % 100) <= 13:
        suffix = 'TH'
    else:
        suffix = ['TH', 'ST', 'ND', 'RD', 'TH'][min(n % 10, 4)]
    return str(n) + suffix + match.group(2)

# Function to update the address calling make_ordinal()
def update_address(address):
    pattern = r'(\d+)(\s+(?:Street|Avenue|Blvd|Drive))'
    result = re.sub(pattern, make_ordinal, address, flags=re.IGNORECASE)
    return result

def download_and_unzip(url, directory_name):
    # Delete the directory if it already exists
    if os.path.exists(directory_name):
        shutil.rmtree(directory_name)

    # Create the directory
    os.makedirs(directory_name)

    # Download the shapefile zip file
    zip_file_path = os.path.join(directory_name, f'{directory_name}.zip')
    response = requests.get(url)
    with open(zip_file_path, 'wb') as f:
        f.write(response.content)

    # Unzip the file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(directory_name)

    # Remove the zip file
    os.remove(zip_file_path)

    print(f"Shapefile downloaded and unzipped in '{directory_name}' directory.")

def load_csv(file_path):
    return pd.read_csv(file_path)