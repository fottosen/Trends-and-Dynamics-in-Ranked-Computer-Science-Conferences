from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd

# Set your SPARQL endpoint
sparql = SPARQLWrapper("https://sparql.dblp.org/sparql")

# Step 1: Get venues with more than 1000 publications
venue_query = """
PREFIX dblp: <https://dblp.org/rdf/schema#>

SELECT ?venue
WHERE {
  ?pub a dblp:Publication .
  ?pub dblp:publishedIn ?venue .
  ?pub a ?type .
  FILTER (?type = dblp:Inproceedings || ?type = dblp:Article)
}
GROUP BY ?venue
HAVING (COUNT(DISTINCT ?pub) > 1000)
"""

sparql.setQuery(venue_query)
sparql.setReturnFormat(JSON)
venue_results = sparql.query().convert()

# Extract venue IRIs and format them correctly
venues = [result["venue"]["value"] for result in venue_results["results"]["bindings"]]

# Step 2: For each venue, get 100 publications
publications = []
for venue in venues:
   
    pub_query = f"""
    PREFIX dblp: <https://dblp.org/rdf/schema#>

    SELECT ?title ?doi
    WHERE {{
      ?pub dblp:publishedIn "{venue}" .
      ?pub dblp:title ?title .
      ?pub dblp:doi ?doi .
      ?pub a dblp:Publication .
      ?pub a ?type .
      FILTER (?type = dblp:Inproceedings || ?type = dblp:Article)
    }}
    
    """
    #LIMIT 100 >> removed from query above

    sparql.setQuery(pub_query)
    sparql.setReturnFormat(JSON)
    
    pub_results = sparql.query().convert()
    
    # Collect the publication data
    for result in pub_results["results"]["bindings"]:
        publications.append({
            "venue": venue,
            "title": result["title"]["value"],
            "doi": result.get("doi", {}).get("value", "N/A")  # Handle cases where DOI may not exist
        })

# Step 3: Create a DataFrame and save to CSV
df = pd.DataFrame(publications)

# Save the DataFrame to a CSV file
csv_file_path = "dblp_publications.csv"
df.to_csv(csv_file_path, index=False, encoding='utf-8')

print(f"CSV file '{csv_file_path}' created with {len(publications)} publications.")


import pandas as pd

import pandas as pd

# Path to your CSV file
csv_file = ''

# Read the CSV file
df = pd.read_csv(csv_file)

# Check if required columns exist
required_columns = {'venue_acronym', 'rank'}
if not required_columns.issubset(df.columns):
    missing = required_columns - set(df.columns)
    print(f"Missing columns in CSV file: {', '.join(missing)}")
else:
    # Function to process each rank
    def process_rank(rank_label):
        subset = df[df['rank'] == rank_label]
        unique_acronyms = subset['venue_acronym'].dropna().unique()
        unique_count = len(unique_acronyms)
        print(f"Number of unique 'venue_acronym' values for rank '{rank_label}': {unique_count}")
        print(f"Unique Acronyms for rank '{rank_label}':")
        for acronym in unique_acronyms:
            print(acronym)
        print("-" * 40)

    # Process for rank 'A'
    process_rank('A')

    # Process for rank 'B'
    process_rank('B')


