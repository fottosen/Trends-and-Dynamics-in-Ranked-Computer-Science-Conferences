import pandas as pd
import requests
import re
import time
import os

"""key = os.getenv("KEY")

# Function to fetch abstracts for a batch of DOIs
def fetch_abstracts_batch(dois):    
    try:
        # Send a POST request with the list of DOIs
        response = requests.post(
            "https://api.semanticscholar.org/graph/v1/paper/batch",
            headers={"x-api-key": key},
            params={"fields": "externalIds,abstract"}, 
            json={"ids": dois}
        )

        if response.status_code == 200:
            data = response.json()
            #print("response.json()", data)
            # Filter out None entries and extract doi and abstract, converting the former to uppercase
            return {
                item['externalIds']['DOI'].upper(): item.get('abstract', "N/A") for item in data if item is not None
            }
        else:
            return {doi: f"Error {response.status_code}: N/A" for doi in dois}

    except Exception as e:
        return {doi: f"Exception: {str(e)}" for doi in dois}

# Load CSV 
df = pd.read_csv('')
#df = pd.read_csv('sampled_file.csv')
#df=df.sample(n=1000, random_state=1)

df['doi'] = df['doi'].apply(lambda x: re.sub(r'^[^#]*?:\/\/[a-zA-Z0-9.-]+\/', '', str(x)))

#df['doi']

# Empty list to store sampled DataFrames for each venue
sampled_dfs = []

excepCount=0

# For each distinct venue, take 
for venue in df['venue_title'].unique():
    venue_df = df[df['venue_title'] == venue]
    
    # Break the DOIs into batches
    dois = venue_df['doi'].tolist()
    batch_size = 100
    venue_df['abstracts'] = "N/A"  # Initialize column for abstracts

    # Fetch abstracts in batches of batch_size
    for i in range(0, len(dois), batch_size):
        print("3")
        batch = dois[i:i+batch_size]
        print(f"Processing batch of {len(batch)} DOIs")
        abstract_data = fetch_abstracts_batch(batch)

        # Assign the fetched abstracts back to the DataFrame
        
        for doi in batch:
            try:
                venue_df.loc[venue_df['doi'] == doi, 'abstracts'] = abstract_data[doi]

            except Exception as e:
                excepCount+=1
                print(f"Exception: {str(e)}")
                continue
        time.sleep(2)  # 1 second delay between batch requests to avoid overloading the sScholarAPI
    
    # Append the sampled DataFrame to the sampled_dfs list
    sampled_dfs.append(venue_df)

# Merge all sampled DataFrames into a single DataFrame
final_df = pd.concat(sampled_dfs)

# Reset the index of the final DataFrame
final_df.reset_index(drop=True, inplace=True)

# Save the final DataFrame to a CSV file
final_df.to_csv('DBLP_A_B_venues_abstracts.csv', index=False)

# Final print for code completion
print("OK")
print("excepCount is: ",excepCount)"""

# API key
key = os.getenv("KEY")

# Timeout in seconds
REQUEST_TIMEOUT = 20  

# Function to fetch abstracts for a batch of DOIs
def fetch_abstracts_batch(dois):
    try:
        # Send a POST request with the list of DOIs and a specified timeout
        response = requests.post(
            "https://api.semanticscholar.org/graph/v1/paper/batch",
            headers={"x-api-key": key},
            params={"fields": "externalIds,abstract"}, 
            json={"ids": dois},
            timeout=REQUEST_TIMEOUT  # 20 seconds timeout
        )

        if response.status_code == 200:
            data = response.json()
            return {
                item['externalIds']['DOI'].upper(): item.get('abstract', "N/A") for item in data if item is not None
            }
        else:
            return {doi: f"Error {response.status_code}: N/A" for doi in dois}
    
    except Exception as e:
        return {doi: f"Exception: {str(e)}" for doi in dois}

# For handling large CSV files
def process_and_save_in_chunks(input_file, output_file, batch_size=100, chunk_size=1000):

    # To handle appending to the results CSV
    first_run = not os.path.exists(output_file)

    chunk_iter = pd.read_csv(input_file, chunksize=chunk_size)
    excep_count = 0
    
    for chunk_num, chunk_df in enumerate(chunk_iter):
        print(f"Processing Chunk {chunk_num + 1}")
        
        # Clean up DOI column
        chunk_df['doi'] = chunk_df['doi'].apply(lambda x: re.sub(r'^[^#]*?:\/\/[a-zA-Z0-9.-]+\/', '', str(x)))
        chunk_df['abstracts'] = "N/A"  # Initialize new column
        
        # Loop over each venue and fetch abstracts in batches
        unique_venues = chunk_df['venue_title'].unique()
        
        for venue in unique_venues:
            venue_df = chunk_df[chunk_df['venue_title'] == venue]
            dois = venue_df['doi'].tolist()
            
            # Fetching abstract in batches (API robustness)
            for i in range(0, len(dois), batch_size):
                sub_batch = dois[i:i + batch_size]
                print(f"Processing batch of {len(sub_batch)} for venue {venue}")

                # Actual API call
                abstracts_data = fetch_abstracts_batch(sub_batch)

                # Backfilling abstracts
                for doi in sub_batch:
                    try:
                        chunk_df.loc[chunk_df['doi'] == doi, 'abstracts'] = abstracts_data[doi]
                    except Exception as e:
                        excep_count += 1
                        print(f"Exception: {str(e)}")
                        continue

                time.sleep(2)  # Avoid API rate-limits 

        # Save the partial results to file after each chunk
        if first_run:
            chunk_df.to_csv(output_file, mode='w', index=False)  # Write new file for first run
            first_run = False
        else:
            chunk_df.to_csv(output_file, mode='a', header=False, index=False)  # Append to file in append mode
        
    print(f"Process completed! Exceptions count: {excep_count}")

# Usage
input_csv = ''
output_csv = ''

# Call the function to start processing in chunks
process_and_save_in_chunks(input_file=input_csv, output_file=output_csv)

















"""# For each distinct venue, take a random sample of k
k=200
for venue in df['venue_title'].unique():
    venue_df = df[df['venue'] == venue].sample(n=k, replace=False, random_state=1)
    
    # Break the DOIs into batches
    dois = venue_df['doi'].tolist()
    batch_size = 500
    venue_df['abstracts'] = None  # Initialize column for abstracts

    # Fetch abstracts in batches of batch_size
    for i in range(0, len(dois), batch_size):
        batch = dois[i:i+batch_size]
        abstract_data = fetch_abstracts_batch(batch)
        
        # Assign the fetched abstracts back to the DataFrame
        for doi in batch:
            venue_df.loc[venue_df['doi'] == doi, 'abstracts'] = abstract_data.get(doi, 'No abstract available')
        
        time.sleep(1)  # 1 second delay between batch requests to avoid overloading the sScholarAPI
    
    # Append the sampled DataFrame to the sampled_dfs list
    sampled_dfs.append(venue_df)

# Merge all sampled DataFrames into a single DataFrame
final_df = pd.concat(sampled_dfs)

# Reset the index of the final DataFrame
final_df.reset_index(drop=True, inplace=True)

# Save the final DataFrame to a CSV file
final_df.to_csv('dblp_topVenues_abstracts.csv', index=False)

# Final print for code completion
print("OK")
"""