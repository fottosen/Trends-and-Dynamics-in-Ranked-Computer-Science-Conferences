import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON
import time

"""
### Give ICORE csv propper headers
df = pd.read_csv("", header=None)
header= ["id","title","acronym","source","rank","6","7","8","9"]
df.columns = header
df.to_csv("ICORE_Conf_Portal_Ranks.csv", index=False)

"""
'''
# Step 1: Load CSV file form ICORE with rank information
ICORE_extract = ""
df_venues = pd.read_csv(ICORE_extract)

def FindAStar():
    # Filter for A* ranked venues
    a_star_venues = df_venues[df_venues["rank"] == "A*"]

    # Replace ICORE acronym with correct DBLP acronym
    a_star_venues['acronym'] = a_star_venues['acronym'].replace('ACMMM', 'ACM Multimedia')
    a_star_venues['acronym'] = a_star_venues['acronym'].replace('EuroCrypt', 'EUROCRYPT')
    a_star_venues['acronym'] = a_star_venues['acronym'].replace('MOBICOM', 'MobiCom')
    a_star_venues['acronym'] = a_star_venues['acronym'].replace('SENSYS', 'SenSys')
    a_star_venues['acronym'] = a_star_venues['acronym'].replace('SIGMOD', 'SIGMOD Conference')


    # Add a alternative acronym to SIGMOD conference
    sigmod_extra_acronym = a_star_venues[a_star_venues['acronym'] == 'SIGMOD Conference'].copy()
    sigmod_extra_acronym['acronym'] = 'SIGMOD Conference Companion'

    # Append the new row to the original a_star_venues DF
    a_star_venues = pd.concat([a_star_venues, sigmod_extra_acronym], ignore_index=True)

    # Set SPARQL endpoint
    sparql = SPARQLWrapper("https://sparql.dblp.org/sparql")

    # Step 2: Query publications for A* venues
    publications = []
    venues_not_found = []  # List to track venues with no results

    for index, row in a_star_venues.iterrows():
        acronym = row["acronym"]
        title = row["title"]
        rank = row["rank"]
        
        # SPARQL query to get publications for the current venue (acronym)
        pub_query = f"""
        PREFIX dblp: <https://dblp.org/rdf/schema#>

        SELECT ?title ?doi ?year
        WHERE {{
        ?pub dblp:publishedIn "{acronym}" .
        ?pub dblp:title ?title .
        ?pub dblp:doi ?doi .
        ?pub a dblp:Publication .
        ?pub a ?type .
        ?pub dblp:yearOfPublication ?year .
        FILTER (?type = dblp:Inproceedings || ?type = dblp:Article)
        }}
        """
        
        sparql.setQuery(pub_query)
        sparql.setReturnFormat(JSON)
        
        try:
            pub_results = sparql.query().convert()
        
            if not pub_results["results"]["bindings"]:
                # If no results were returned for this venue, add it to venues_not_found
                venues_not_found.append(acronym)
                print(f"No publications found for venue {acronym}.")
            else:
                # Collect the publication data and add venue title and rank
                for result in pub_results["results"]["bindings"]:
                    publications.append({
                        "venue_acronym": acronym,
                        "venue_title": title,
                        "rank": rank,
                        "publication_title": result["title"]["value"],
                        "year": result["year"]["value"],
                        "doi": result["doi"]["value"]
                    })
        
        except Exception as e:
            print(f"Error querying venue {acronym}: {e}")

    # Step 3: Create a DataFrame from the publication results
    df_publications = pd.DataFrame(publications)
    return df_publications

#df_publications = FindAStar()

## Save the DataFrame to a new CSV file
#output_csv_file_path = "DBLP_a_star_venues.csv"
#df_publications.to_csv(output_csv_file_path, index=False, encoding='utf-8')

def FindA_B():
    # Filter for A* ranked venues
    a_b_venues = df_venues[(df_venues["rank"] == "A") | (df_venues["rank"] == "B")]

    # Replace ICORE acronym with correct DBLP acronym
    a_b_venues['acronym'] = a_b_venues['acronym'].replace('SIGSPATIAL', 'SIGSPATIAL/GIS') #GIS
    a_b_venues['acronym'] = a_b_venues['acronym'].replace('IEEE VIS', 'VIS')
    a_b_venues['acronym'] = a_b_venues['acronym'].replace('MOBIHOC', 'MobiHoc')
    a_b_venues['acronym'] = a_b_venues['acronym'].replace('MSWIM', 'MSWiM')
    a_b_venues['acronym'] = a_b_venues['acronym'].replace('ECML PKDD', 'PKDD')
    a_b_venues['acronym'] = a_b_venues['acronym'].replace('Mobisys', 'MobiSys')
    a_b_venues['acronym'] = a_b_venues['acronym'].replace('CaiSE', 'CAiSE')
    a_b_venues['acronym'] = a_b_venues['acronym'].replace('FOSSACS', 'FoSSaCS')
    a_b_venues['acronym'] = a_b_venues['acronym'].replace('IEEE CCNC', 'CCNC')
    a_b_venues['acronym'] = a_b_venues['acronym'].replace('IEEE-IV', 'IV')
    a_b_venues['acronym'] = a_b_venues['acronym'].replace('PKC', 'Public Key Cryptography')
    a_b_venues['acronym'] = a_b_venues['acronym'].replace('EUROGP', 'EuroGP')
    a_b_venues['acronym'] = a_b_venues['acronym'].replace('EuroPar', 'Euro-Par')
    a_b_venues['acronym'] = a_b_venues['acronym'].replace('I3DG', 'I3D') #SI3D
    a_b_venues['acronym'] = a_b_venues['acronym'].replace('ACM_WiSec', 'WISEC')
    a_b_venues['acronym'] = a_b_venues['acronym'].replace('SoCS', 'SOCS')
    a_b_venues['acronym'] = a_b_venues['acronym'].replace('IEEE SSE', 'IEEE SCC')
    a_b_venues['acronym'] = a_b_venues['acronym'].replace('SGP', 'Symposium on Geometry Processing')
    a_b_venues['acronym'] = a_b_venues['acronym'].replace('Group', 'GROUP')

    

    # Add a alternative acronym to SIGSPATIAL/GIS conference
    SIGSPATIAL_extra_acronym = a_b_venues[a_b_venues['acronym'] == 'SIGSPATIAL/GIS'].copy()
    SIGSPATIAL_extra_acronym['acronym'] = 'GIS'
    # Append the new row to the original a_b_venues DF
    a_b_venues = pd.concat([a_b_venues, SIGSPATIAL_extra_acronym], ignore_index=True)

    # Add a alternative acronym to I3D conference
    I3D_extra_acronym = a_b_venues[a_b_venues['acronym'] == 'I3D'].copy()
    I3D_extra_acronym['acronym'] = 'SI3D'
    # Append the new row to the original a_b_venues DF
    a_b_venues = pd.concat([a_b_venues, I3D_extra_acronym], ignore_index=True)

    # Set SPARQL endpoint
    sparql = SPARQLWrapper("https://sparql.dblp.org/sparql")

    # Step 2: Query publications for A* venues
    publications = []
    venues_not_found = []  # List to track venues with no results

    for index, row in a_b_venues.iterrows():
        acronym = row["acronym"]
        title = row["title"]
        rank = row["rank"]
        
        # SPARQL query to get publications for the current venue (acronym)
        pub_query = f"""
        PREFIX dblp: <https://dblp.org/rdf/schema#>

        SELECT ?title ?doi ?year
        WHERE {{
        ?pub dblp:publishedIn "{acronym}" .
        ?pub dblp:title ?title .
        ?pub dblp:doi ?doi .
        ?pub a dblp:Publication .
        ?pub a ?type .
        ?pub dblp:yearOfPublication ?year .
        FILTER (?type = dblp:Inproceedings || ?type = dblp:Article)
        }}
        """
        
        sparql.setQuery(pub_query)
        sparql.setReturnFormat(JSON)
        
        try:
            pub_results = sparql.query().convert()
        
            if not pub_results["results"]["bindings"]:
                # If no results were returned for this venue, add it to venues_not_found
                venues_not_found.append(acronym)
                print(f"No publications found for venue {acronym}.")
            else:
                # Collect the publication data and add venue title and rank
                for result in pub_results["results"]["bindings"]:
                    publications.append({
                        "venue_acronym": acronym,
                        "venue_title": title,
                        "rank": rank,
                        "publication_title": result["title"]["value"],
                        "year": result["year"]["value"],
                        "doi": result["doi"]["value"]
                    })
        
        except Exception as e:
            print(f"Error querying venue {acronym}: {e}")

    # Step 3: Create a DataFrame from the publication results
    df_publications = pd.DataFrame(publications)
    return df_publications

# df = FindA_B()

# # Save the DataFrame to a new CSV file
# output_csv_file_path = "DBLP_A_B_venues.csv"
# df.to_csv(output_csv_file_path, index=False, encoding='utf-8')
'''
def FindC():
    # Filter for C ranked venues
    c_venues = df_venues[(df_venues["rank"] == "C")]

    # Replace ICORE acronym with correct DBLP acronym
    c_venues['acronym'] = c_venues['acronym'].replace('SASHIMI', 'SASHIMI@MICCAI')
    c_venues['acronym'] = c_venues['acronym'].replace('Tencon', 'TENCON')
    c_venues['acronym'] = c_venues['acronym'].replace('COG', 'CoG')
    c_venues['acronym'] = c_venues['acronym'].replace('Coordination', 'COORDINATION')
    c_venues['acronym'] = c_venues['acronym'].replace('CSEET', 'CSEE&T')
    c_venues['acronym'] = c_venues['acronym'].replace('DIGRA', 'DiGRA')
    c_venues['acronym'] = c_venues['acronym'].replace('Euro VR', 'EuroXR')
    c_venues['acronym'] = c_venues['acronym'].replace('EUSFLAT', 'EUSFLAT Conf.')
    c_venues['acronym'] = c_venues['acronym'].replace('HASKELL', 'Haskell')
    c_venues['acronym'] = c_venues['acronym'].replace('HOTCHIPS (HCS)', 'HCS')
    c_venues['acronym'] = c_venues['acronym'].replace('ICCSA', 'ICCSA (1)')
    c_venues['acronym'] = c_venues['acronym'].replace('IEEE CIFEr', 'CIFEr')
    c_venues['acronym'] = c_venues['acronym'].replace('IEEE IS', 'IS')
    c_venues['acronym'] = c_venues['acronym'].replace('IIWAS', 'iiWAS')
    c_venues['acronym'] = c_venues['acronym'].replace('IWSM Mensura', 'IWSM/Mensura')
    c_venues['acronym'] = c_venues['acronym'].replace('IJCRS (was RSCTC)', 'IJCRS')
    c_venues['acronym'] = c_venues['acronym'].replace('OPENSYM', 'OpenSym')
    c_venues['acronym'] = c_venues['acronym'].replace('Qshine', 'QSHINE')
    c_venues['acronym'] = c_venues['acronym'].replace('ICSoft', 'ICSOFT')
    c_venues['acronym'] = c_venues['acronym'].replace('Mobiquitous', 'MobiQuitous')
    c_venues['acronym'] = c_venues['acronym'].replace('Broadnets', 'BROADNETS')
    c_venues['acronym'] = c_venues['acronym'].replace('DIAGRAMS', 'Diagrams')
    c_venues['acronym'] = c_venues['acronym'].replace('CICLING', 'CICLing')
    c_venues['acronym'] = c_venues['acronym'].replace('DAFX', 'DAFx')
    c_venues['acronym'] = c_venues['acronym'].replace('APWEB', 'APWeb')
    c_venues['acronym'] = c_venues['acronym'].replace('MOMM', 'MoMM')
    c_venues['acronym'] = c_venues['acronym'].replace('Algosensors', 'ALGOSENSORS')
        
    # Add a alternative acronym to IJCRS conference
    IJCRS_extra_acronym = c_venues[c_venues['acronym'] == 'IJCRS'].copy()
    IJCRS_extra_acronym['acronym'] = 'RSCTC'
    # Append the new row to the original c_venues DF
    c_venues = pd.concat([c_venues, IJCRS_extra_acronym], ignore_index=True)

    # Set SPARQL endpoint
    sparql = SPARQLWrapper("https://sparql.dblp.org/sparql")

    # Step 2: Query publications for C venues
    publications = []
    venues_not_found = []  # List to track venues with no results

    for index, row in c_venues.iterrows():
        acronym = row["acronym"]
        title = row["title"]
        rank = row["rank"]
        
        # SPARQL query to get publications for the current venue (acronym)
        pub_query = f"""
        PREFIX dblp: <https://dblp.org/rdf/schema#>

        SELECT ?title ?doi ?year
        WHERE {{
        ?pub dblp:publishedIn "{acronym}" .
        ?pub dblp:title ?title .
        ?pub dblp:doi ?doi .
        ?pub a dblp:Publication .
        ?pub a ?type .
        ?pub dblp:yearOfPublication ?year .
        FILTER (?type = dblp:Inproceedings || ?type = dblp:Article)
        }}
        """
        
        sparql.setQuery(pub_query)
        sparql.setReturnFormat(JSON)
        
        try:
            pub_results = sparql.query().convert()
        
            if not pub_results["results"]["bindings"]:
                # If no results were returned for this venue, add it to venues_not_found
                venues_not_found.append(acronym)
                print(f"No publications found for venue {acronym}.")
            else:
                # Collect the publication data and add venue title and rank
                for result in pub_results["results"]["bindings"]:
                    publications.append({
                        "venue_acronym": acronym,
                        "venue_title": title,
                        "rank": rank,
                        "publication_title": result["title"]["value"],
                        "year": result["year"]["value"],
                        "doi": result["doi"]["value"]
                    })
        
        except Exception as e:
            print(f"Error querying venue {acronym}: {e}")

    # Step 3: Create a DataFrame from the publication results
    df_publications = pd.DataFrame(publications)
    return df_publications

# df = FindC()

# # Save the DataFrame to a new CSV file
# output_csv_file_path = "DBLP_C_venues.csv"
# df.to_csv(output_csv_file_path, index=False, encoding='utf-8')
'''

def FindAllInproceedings():
    # Set SPARQL endpoint
    sparql = SPARQLWrapper("https://sparql.dblp.org/sparql")

    # Base query template with double curly braces for SPARQL placeholders
    query_template = """
    PREFIX dblp: <https://dblp.org/rdf/schema#>

    SELECT ?title ?doi ?year ?venue ?numberOfCreators
    WHERE {{
        ?pub a dblp:Inproceedings .
        ?pub dblp:title ?title .
        ?pub dblp:doi ?doi . 
        ?pub dblp:yearOfPublication ?year . 
        ?pub dblp:publishedIn ?venue .
        OPTIONAL {{ ?pub dblp:numberOfCreators ?numberOfCreators . }}
    }}
    LIMIT 10000
    OFFSET {offset}
    """

    publications = []
    offset = 0
    chunk_size = 10000  # Fetch 10,000 records at a time
    has_more_results = True

    while has_more_results:
        # Update the query with the current offset
        pub_query = query_template.format(offset=offset)

        sparql.setQuery(pub_query)
        sparql.setReturnFormat(JSON)

        try:
            # Execute the query
            pub_results = sparql.query().convert()

            # Check if any results were returned
            if not pub_results["results"]["bindings"]:
                has_more_results = False  # Stop if no more results are found
                break

            # Parse the results and append them to the publications list
            for result in pub_results["results"]["bindings"]:
                publication = {
                    "publication_title": result["title"]["value"]
                    ,
                    "doi": result["doi"]["value"]# if "doi" in result else None
                    ,
                    "year": result["year"]["value"]# if "year" in result else None
                    ,
                    "venue_acronym": result["venue"]["value"]# if "venue" in result else None
                    ,
                    "numberOfCreators": result["numberOfCreators"]["value"] if "numberOfCreators" in result else None
                }
                publications.append(publication)

            # Update the offset for the next chunk
            offset += chunk_size

            print(f"Retrieved {len(publications)} publications so far...")

        except Exception as e:
            print(f"Error querying publications with offset {offset}: {e}")
            break  # Stop if there is an error

    # Step 3: Create a DataFrame from the publication results
    df_publications = pd.DataFrame(publications)
    return df_publications

df = FindAllInproceedings()

# # Save the DataFrame to a new CSV file
# output_csv_file_path = ""
# df.to_csv(output_csv_file_path, index=False, encoding='utf-8')


### All authors in DBLP Icore ranks:
'''
import concurrent.futures
# Load the CSV file
df_dois = pd.read_csv("")

# Ensure 'doi' column exists
if 'doi' not in df_dois.columns:
    raise ValueError("'doi' column not found in the input CSV")

# SPARQL query function
def query_dblp_for_authors(doi):
    sparql = SPARQLWrapper("https://sparql.dblp.org/sparql")
    query = f"""
    PREFIX dblp: <https://dblp.org/rdf/schema#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    
    SELECT ?author ?wikiq ?name WHERE {{
      ?publ dblp:doi <https://doi.org/{doi}> .
      ?publ dblp:authoredBy ?author .
      ?author dblp:primaryCreatorName ?name .
      OPTIONAL {{
        ?author dblp:wikidata ?wikiq .
      }}
    }}
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    
    try:
        results = sparql.query().convert()
        author_data = []
        for result in results["results"]["bindings"]:
            dblp_pid = result["author"]["value"] if "author" in result else None
            wiki_qid = result["wikiq"]["value"] if "wikiq" in result else None
            name = result["name"]["value"] if "name" in result else None
            
            author_data.append({
                "doi": doi,
                "dblpPID": dblp_pid,
                "wikiP": wiki_qid,
                "name": name
            })
        return author_data
    
    except Exception as e:
        print(f"Error querying for DOI {doi}: {e}")
        return []

# Function to process multiple DOIs in parallel
def process_dois_in_parallel(dois, max_workers=10):
    all_author_data = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_doi = {executor.submit(query_dblp_for_authors, doi): doi for doi in dois}
        for future in concurrent.futures.as_completed(future_to_doi):
            doi = future_to_doi[future]
            try:
                result = future.result()
                all_author_data.extend(result)
            except Exception as e:
                print(f"Error processing DOI {doi}: {e}")
    return all_author_data

# Start the total timing
start_total_time = time.time()

# Fetch author details for all DOIs in parallel
all_author_data = process_dois_in_parallel(df_dois['doi'], max_workers=10)

# Save the results to a new CSV file
output_csv_file_path = ""
df_output = pd.DataFrame(all_author_data)
df_output.to_csv(output_csv_file_path, index=False, encoding='utf-8')

print(f"Results saved to {output_csv_file_path}")
end_query_time = time.time()
print(f"Time taken: {end_query_time - start_total_time:.2f} seconds")
'''


# Find all authors
'''
# Step 1: Load the CSV file with DOIs
df_dois = pd.read_csv("")

# Ensure 'doi' column exists
if 'doi' not in df_dois.columns:
    raise ValueError("'doi' column not found in the input CSV")

# Step 2: Set up the SPARQL query
def query_dblp_for_authors(doi):
    # Initialize SPARQL wrapper
    sparql = SPARQLWrapper("https://sparql.dblp.org/sparql")
    
    # Prepare the query by inserting the DOI
    query = f"""
    PREFIX dblp: <https://dblp.org/rdf/schema#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    
    SELECT ?author ?wikiq ?name WHERE {{
      ?publ dblp:doi <https://doi.org/{doi}> .
      ?publ dblp:authoredBy ?author .
      ?author dblp:primaryCreatorName ?name .
      OPTIONAL {{
        ?author dblp:wikidata ?wikiq .
      }}
    }}
    """
    
    # Execute the query
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    
    try:
        # Fetch results
        results = sparql.query().convert()
        
        # Process results into a list of dictionaries
        author_data = []
        for result in results["results"]["bindings"]:
            dblp_pid = result["author"]["value"] if "author" in result else None
            wiki_qid = result["wikiq"]["value"] if "wikiq" in result else None
            name = result["name"]["value"] if "name" in result else None
            
            author_data.append({
                "doi": doi,
                "dblpPID": dblp_pid,
                "wikiP": wiki_qid,
                "name": name
            })
        
        return author_data
    
    except Exception as e:
        print(f"Error querying for DOI {doi}: {e}")
        return []  # Return empty if error

# Step 3: Fetch author details for all DOIs
all_author_data = []

# Start the total timing
start_total_time = time.time()

for doi in df_dois['doi'][0:8000]:
    doi_author_data = query_dblp_for_authors(doi)
    all_author_data.extend(doi_author_data)  # Append the results

# Step 4: Save the results to a new CSV file
output_csv_file_path = ""
df_output = pd.DataFrame(all_author_data)
df_output.to_csv(output_csv_file_path, index=False, encoding='utf-8')

print(f"Results saved to {output_csv_file_path}")
end_query_time = time.time()
print(f"Time taken to query DBLP for 8000 DOIs: {end_query_time - start_total_time:.2f} seconds")


