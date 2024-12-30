"""
###

    TEST

###

import requests

DOI = "10.23919/ACC53348.2022.9867534" 
url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{DOI}?fields=abstract"


response = requests.get(url)
data = response.json()
data2 = {}

print(data.get("abstract", "No abstract available"))

"""


import pandas as pd
import requests
import re
import time

# Function to fetch abstract using DOI
def fetch_abstract(doi):
    # Use regex to remove "https://doi.org/" from the DOI
    cleaned_doi = re.sub(r'^[^#]*?:\/\/[a-zA-Z0-9.-]+\/', '', doi)
    url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{cleaned_doi}?fields=abstract"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            # Return the abstract if available, else return 'No abstract available'
            return data.get("abstract", "No abstract available")
        else:
            return f"Error {response.status_code}: Unable to fetch abstract"
    except Exception as e:
        return f"Exception: {str(e)}"

# Sleep method - 1s per request
def process_doi(doi):
    # Pause for 1 second before each request
    time.sleep(1)
    return fetch_abstract(doi)

# Start the timer
start_time = time.time()

# Read the CSV file with pandas
input_csv = "ssize_100_ACC_inpros.csv"  
output_csv = input_csv + "with_abstracts.csv"  # Output CSV file with abstracts

# Load CSV into a DataFrame
df = pd.read_csv(input_csv)

# Apply the process_doi function to fetch abstract with/out a 1-second pause
df['abstract'] = df['doi'].apply(fetch_abstract)

# Save the updated DataFrame to a new CSV file
df.to_csv(output_csv, index=False)

# Stop the timer and calculate the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time

print(f"Processed CSV saved to {output_csv}")
print(f"Time taken: {elapsed_time} seconds")









def testAB(doi):
    return "TEST "+doi


import pandas as pd

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('dblp_top1000venue_doi.csv')

# Create an empty list to store sampled DataFrames
sampled_dfs = []


x=0
# For each distinct venue, take a random sample of 100 rows
for venue in df['venue'].unique():
    venue_df = df[df['venue'] == venue].sample(n=100, replace=False, random_state=1)
    
    # Add a new column "abstracts" (you can fill it with placeholders or actual data)
    venue_df['abstracts'] = df['doi'].apply(testAB)
    
    # Append the sampled DataFrame to the list
    sampled_dfs.append(venue_df)

# Merge all sampled DataFrames into a single DataFrame
final_df = pd.concat(sampled_dfs)

# Reset the index of the final DataFrame
final_df.reset_index(drop=True, inplace=True)

# Save the final DataFrame to a CSV file (optional)
final_df.to_csv('sampled_venues.csv', index=False)

print("OK")

