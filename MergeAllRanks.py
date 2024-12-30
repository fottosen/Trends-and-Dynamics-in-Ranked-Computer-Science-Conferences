import re
import pandas as pd

# Merge ranks together
'''
# Read the CSV files
df_a_star = pd.read_csv("CSO_Mapping/DBLP_a_star_venues_w_topics.csv")
df_A = pd.read_csv("CSO_Mapping/DBLP_A_venues_w_topics.csv")
df_B = pd.read_csv("CSO_Mapping/DBLP_B_venues_w_topics.csv")
df_C = pd.read_csv("CSO_Mapping/DBLP_C_venues_w_topics.csv")

# Add the rank column to each DataFrame
df_a_star['rank'] = 'A*'
df_A['rank'] = 'A'
df_B['rank'] = 'B'
df_C['rank'] = 'C'

# Concatenate the DataFrames
merged_df = pd.concat([df_a_star, df_A, df_B, df_C], ignore_index=True)

# Optionally, save the merged DataFrame to a new CSV file
merged_df.to_csv("DBLP_w_topics.csv", index=False)

'''

# Check total DBLP inproceedings vs already processed inproceedings
'''
topic_df = pd.read_csv('DBLP_w_topics.csv')
inproceedings_df = pd.read_csv('DBLP_all_inproceddings.csv')

# Step 2: Extract DOIs from the 'topic' file
# Assuming the DOI column is named 'doi' in the topic file
dois_to_filter = topic_df['doi'].unique()
inproceedings_df = inproceedings_df.dropna(subset=['doi'])
inproceedings_df['doi'] = inproceedings_df['doi'].apply(lambda x: re.sub(r'^[^#]*?:\/\/[a-zA-Z0-9.-]+\/', '', str(x)))

# Step 3: Filter the rows in 'all_inproceedings' that match the DOIs
# Assuming the DOI column is also named 'doi' in the all_inproceedings file
filtered_inproceedings = inproceedings_df[~inproceedings_df['doi'].isin(dois_to_filter)]

# Step 4: Save the filtered result to a new CSV file
filtered_inproceedings.to_csv('DBLP_inproceedings_rest.csv', index=False)

print("Filtering completed. Results saved to 'filtered_inproceedings.csv'.")


test = pd.read_csv('DBLP_inproceedings_rest.csv')

'''


# Add number of authors from DBLP to data
'''
df_w_topics = pd.read_csv('DBLP_w_topics.csv')
df_DBLP_all = pd.read_csv('abstract_API/generate_CSV/DBLP_all_inproceddings.csv')
df_DBLP_all['doi'] = df_DBLP_all['doi'].apply(lambda x: re.sub(r'^[^#]*?:\/\/[a-zA-Z0-9.-]+\/', '', str(x)))

# Merge the dataframes on the 'doi' column
df_merged = pd.merge(df_w_topics, df_DBLP_all[['doi', 'numberOfCreators']], on='doi', how='left')

# Save the enriched dataframe to a new CSV file
df_merged.to_csv('df_w_topics_#ofAuthors.csv', index=False)

print("Enrichment complete. The new file 'DBLP_enriched.csv' has been created.")
'''


# Combine wikidata into one df
# Add wikidata gender information to DBLP author csv file
'''
# Define the path to your CSV files
query_files = ['wikiDataGender/query1.csv', 'wikiDataGender/query2.csv', 'wikiDataGender/query3.csv', 'wikiDataGender/query4.csv', 'wikiDataGender/query5.csv']
dblp_file = 'wikiDataGender/DBLP_authors.csv'

# Load and concatenate all query files into one dataframe
df_list = [pd.read_csv(file) for file in query_files]
merged_df = pd.concat(df_list, ignore_index=True)

contains_value = merged_df['wikiq'].str.contains('Q62036606').any()
contains_value2 = merged_df['wikiq'].str.contains('Q5696759').any()

# Load the DBLP authors CSV
dblp_authors_df = pd.read_csv(dblp_file)

# Merge the 'genderLabel' from merged_df into the dblp_authors_df based on the wikiP and wikiq columns
final_df = dblp_authors_df.merge(merged_df[['wikiq', 'genderLabel']], left_on='wikiP', right_on='wikiq', how='left')

# Drop the duplicate 'wikiq' column after merging
final_df.drop('wikiq', axis=1, inplace=True)

# Save the merged dataframe to a new CSV file
final_df.to_csv('wikiDataGender/DBLP_authors_with_gender.csv', index=False)

print("Merged CSV has been saved as 'DBLP_authors_with_gender.csv'.")
'''

# Find gender using wikiq from DBLP 
# or by using name
import time
import pandas as pd
import concurrent.futures
from SPARQLWrapper import SPARQLWrapper, JSON, SPARQLExceptions
import requests

# Set a custom User-Agent for compliance
USER_AGENT = "MSc thesis, catagorizing CS over time (nieo@itu.dk)"

# Wikidata SPARQL query function for gender information
def query_wikidata_for_gender(name):
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.addCustomHttpHeader("User-Agent", USER_AGENT)
    
    '''
    # First query by dblp_id
    query_by_dblp = f"""
    SELECT ?wikiq ?dblpID ?gender ?genderLabel WHERE {{
      ?wikiq wdt:P2456 "{dblp_id}" ;
             wdt:P21 ?gender .
      ?gender rdfs:label ?genderLabel .
      FILTER((LANG(?genderLabel)) = "en")
    }}
    """
    
    sparql.setQuery(query_by_dblp)
    sparql.setReturnFormat(JSON)

    try:
        results = sparql.query().convert()
        gender_data = []
        for result in results["results"]["bindings"]:
            wikiq = result["wikiq"]["value"] if "wikiq" in result else None
            gender_label = result["genderLabel"]["value"] if "genderLabel" in result else None

            gender_data.append({
                "wikiq": wikiq,
                "dblpPID": dblp_id,
                "genderLabel": gender_label
            })
        
        # If results were successfully fetched, return them
        if gender_data:
            return gender_data

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            retry_after = e.response.headers.get("Retry-After", 1)
            print(f"Throttled with HTTP 429. Retrying after {retry_after} seconds...")
            time.sleep(int(retry_after))
            return query_wikidata_for_gender(dblp_id, name)
        else:
            print(f"Error querying for DBLP ID {dblp_id}: {e}")

    except Exception as e:
        print(f"Error querying for DBLP ID {dblp_id}: {e}")
    
    '''
    # If querying by DBLP ID fails, attempt to query by name
    if name:
        query_by_name = f"""
        SELECT ?wikiq ?gender ?genderLabel WHERE {{
          ?wikiq wdt:P31 wd:Q5 ;
                 rdfs:label "{name}"@en ;
                 wdt:P21 ?gender .
          ?gender rdfs:label ?genderLabel .
          FILTER((LANG(?genderLabel)) = "en")
        }}
        """

        sparql.setQuery(query_by_name)
        sparql.setReturnFormat(JSON)

        try:
            results = sparql.query().convert()
            # Ensure we have results and bindings before accessing them
            if "results" in results and "bindings" in results["results"]:
                gender_data = []
                for result in results["results"]["bindings"]:
                    wikiq = result["wikiq"]["value"] if "wikiq" in result else None
                    gender_label = result["genderLabel"]["value"] if "genderLabel" in result else None

                    gender_data.append({
                        "wikiq": wikiq,
                        "dblpPID": None,  
                        "name": name,
                        "genderLabel": gender_label
                    })
                return gender_data
            else:
                print(f"No results found for {name}")
                return []
        

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                retry_after = e.response.headers.get("Retry-After", 1)
                print(f"Throttled with HTTP 429. Retrying after {retry_after} seconds...")
                time.sleep(int(retry_after))
                return query_wikidata_for_gender(name)
            else:
                print(f"Error querying for name {name}: {e}")
        except Exception as e:
            print(f"Error querying for name {name}: {e}")
    return []

# Function to process multiple DBLP IDs in parallel with batching and throttling
def process_dblp_ids_in_parallel(dblp_ids, max_workers):
    all_gender_data = []
    
    # Use ThreadPoolExecutor for parallel processing with a max of 5 threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_dblp_id = {
            executor.submit(query_wikidata_for_gender,dblp_id): dblp_id 
            for dblp_id in dblp_ids
        }
        for index, future in enumerate(concurrent.futures.as_completed(future_to_dblp_id)):
            dblp_id = future_to_dblp_id[future]
            try:
                result = future.result()
                all_gender_data.extend(result)
            except Exception as e:
                print(f"Error processing DBLP ID {dblp_id}: {e}")
    return all_gender_data

###

DBLP_authors = pd.read_csv("wikiDataGender/DBLP_authors.csv")
DBLP_authors_with_gender = pd.read_csv('wikiDataGender/DBLP_authors_with_gender1_2.csv')

# Extract the "name" column from both DataFrames
dblp_names = DBLP_authors['name']
dblp_with_gender_names = DBLP_authors_with_gender['name']

# Find unique names from DBLP_authors that are not in DBLP_authors_with_gender1_2
unique_names = dblp_names[~dblp_names.isin(dblp_with_gender_names)].unique()

# Create a pandas Series from the unique names
unique_names_series = pd.Series(unique_names)

####

# Start the total timing
start_total_time = time.time()

# Fetch gender details for all cleaned DBLP IDs in parallel
all_gender_data = process_dblp_ids_in_parallel(unique_names_series, max_workers=5)

# Save the results to a new CSV file
output_csv_file_path = "DBLP_authors_with_gender_nameGen.csv"
df_output = pd.DataFrame(all_gender_data)
df_output.to_csv(output_csv_file_path, index=False, encoding='utf-8')

print(f"Results saved to {output_csv_file_path}")
end_query_time = time.time()
print(f"Time taken: {end_query_time - start_total_time:.2f} seconds")

