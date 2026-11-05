import os

import requests
from dotenv import load_dotenv


# Load environment variables from the .env file
load_dotenv()

# Read the ESS user ID from the environment
user_id = os.getenv("ESS_USER_ID")

# Define the ESS Round 11 integrated dataset DOI
doi_prefix = "10.21338"
doi_suffix = "ess11e04_2"

# Build the official ESS API endpoint
api_url = (
    f"https://api.ess.sikt.no/v1/data/dataFile/"
    f"{doi_prefix}/{doi_suffix}"
)

# Check that the ESS user ID exists
if not user_id:
    raise ValueError("ESS_USER_ID was not found in the .env file.")

# Define request parameters
params = {
    "userId": user_id,
    "fileFormat": "parquet"
}

print("=" * 70)
print("ESS API CONNECTION TEST")
print("=" * 70)

# Send a streaming request so the full dataset is not loaded into memory
response = requests.get(
    api_url,
    params=params,
    stream=True,
    timeout=60
)

# Stop if the API returned an error
response.raise_for_status()

print(f"HTTP status: {response.status_code}")
print(f"Content type: {response.headers.get('Content-Type')}")
print("ESS API connection successful.")

response.close()