import hashlib
import os
from pathlib import Path

import requests
from dotenv import load_dotenv


# Define the project root directory
project_root = Path(__file__).resolve().parents[1]

# Load environment variables
load_dotenv(project_root / ".env")

# Read the ESS user ID
user_id = os.getenv("ESS_USER_ID")

# Define the ESS Round 11 integrated dataset
doi_prefix = "10.21338"
doi_suffix = "ess11e04_2"

# Build the official ESS API endpoint
api_url = (
    f"https://api.ess.sikt.no/v1/data/dataFile/"
    f"{doi_prefix}/{doi_suffix}"
)

# Define the raw data directory
raw_dir = project_root / "data" / "raw" / "ess"

# Define the output file
output_file = raw_dir / "ESS11e04_2.parquet"

# Define a temporary file used during download
temp_file = raw_dir / "ESS11e04_2.parquet.part"

# Check that the ESS user ID exists
if not user_id:
    raise ValueError("ESS_USER_ID was not found in the .env file.")

# Protect the existing RAW file from accidental overwrite
if output_file.exists():
    raise FileExistsError(
        f"RAW file already exists and will not be overwritten: {output_file}"
    )

# Define request parameters
params = {
    "userId": user_id,
    "fileFormat": "parquet"
}

print("=" * 80)
print("ESS ROUND 11 RAW INGESTION")
print("=" * 80)

print(f"Dataset: ESS Round 11 - Edition 4.2")
print(f"DOI: {doi_prefix}/{doi_suffix}")
print(f"Output: {output_file}")
print()
print("Downloading...")

# Create a SHA-256 checksum during download
sha256 = hashlib.sha256()
total_bytes = 0

# Download the original ESS file
with requests.get(
    api_url,
    params=params,
    stream=True,
    timeout=(30, 300)
) as response:

    # Stop if the API returned an error
    response.raise_for_status()

    with open(temp_file, "wb") as file:

        for chunk in response.iter_content(chunk_size=1024 * 1024):

            if chunk:
                file.write(chunk)
                sha256.update(chunk)
                total_bytes += len(chunk)

# Rename the temporary file after successful download
temp_file.replace(output_file)

# Convert file size to megabytes
file_size_mb = total_bytes / (1024 * 1024)

print()
print("Download completed successfully.")
print(f"File size: {file_size_mb:.2f} MB")
print(f"SHA-256: {sha256.hexdigest()}")
print()
print("RAW ESS file saved without transformations.")