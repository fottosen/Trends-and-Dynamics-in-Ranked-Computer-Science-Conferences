from cso_classifier import CSOClassifier
import pandas as pd
import time

# Load your CSV into a DataFrame
df_main = pd.read_csv('')

#df_sample = df_main.sample(n=2000, random_state=2)

# # # Filter rows where rank is 'A'
# df_A = df_main[df_main['rank'] == 'A']

# # # Randomly sample 200.000 rows where rank is 'B'
#df_B = df_main[df_main['rank'] == 'B']

def process_papers(df):
    # Remove rows where "abstract" is null or has the value 'N/A'
    df_filtered = df[(df['abstracts'].notnull()) & (df['abstracts'] != 'N/A') & (df['abstracts'] != 'No abstracts available')]

    # Create the dictionary `paper`
    papers = {}
    for _, row in df_filtered.iterrows():
        doi = row['doi']
        papers[doi] = {
            "title": row['publication_title'],
            "abstract": row['abstracts']
        }

    start_time = time.time()

    cc = CSOClassifier(workers = 4, modules = "both", enhancement = "all",
                        explanation = False, delete_outliers=True,
                        fast_classification = True, silent = False)
    result = cc.batch_run(papers)


    df_filtered['enhanced'] = None
    df_filtered['syntactic'] = None
    df_filtered['union'] = None

    # Populate the correct row in df_filtered by matching DOI
    for doi, classification in result.items():
        if doi in df_filtered['doi'].values:

            enhanced = ', '.join(classification.get('enhanced', [])) 
            syntactic = ', '.join(classification.get('syntactic', [])) 
            union = ', '.join(classification.get('union', [])) 

            df_filtered.loc[df_filtered['doi'] == doi, 'enhanced'] = enhanced
            df_filtered.loc[df_filtered['doi'] == doi, 'syntactic'] = syntactic
            df_filtered.loc[df_filtered['doi'] == doi, 'union'] = union

    # Total elapsed time
    total_time = time.time()
    print(f"Running the classifier on:\n{len(df_filtered.index)} number of rows took:\n{total_time - start_time:.2f} seconds")

    #df_filtered.to_csv('DBLP_a_star_venues_w_topics.csv', index=False)
    return df_filtered

def process_papers_optimised(df,output_file_path):
    start_time = time.time()
    # Step 1: Remove rows where 'abstracts' is null or 'N/A'
    df_filtered = df[(df['abstracts'].notnull()) & (df['abstracts'] != 'N/A')& (df['abstracts'] != 'No abstracts available')].copy()

    # Step 1.1: Remove any duplicates (issue for rank C)
    df_filtered = df_filtered.drop_duplicates(subset='doi')

    # Step 2: Create the papers dictionary
    papers = df_filtered.set_index('doi')[['publication_title', 'abstracts']].to_dict(orient='index')
    papers = {doi: {"title": info["publication_title"], "abstract": info["abstracts"]}
              for doi, info in papers.items()}

    cc = CSOClassifier(workers=4, modules="both", enhancement="all")

    result = cc.batch_run(papers)

    # Step 4: Add result to df
    df_filtered['enhanced'] = df_filtered['doi'].map(lambda doi: ', '.join(result[doi].get('enhanced', [])) if doi in result else None)
    df_filtered['syntactic'] = df_filtered['doi'].map(lambda doi: ', '.join(result[doi].get('syntactic', [])) if doi in result else None)
    df_filtered['union'] = df_filtered['doi'].map(lambda doi: ', '.join(result[doi].get('union', [])) if doi in result else None)

    # Total elapsed time
    total_time = time.time()
    print(f"Running the classifier on:\n{len(df_filtered.index)} number of rows took:\n{total_time - start_time:.2f} seconds")

    # Save the DataFrame to a new CSV file
    df_filtered.to_csv(output_file_path, index=False, encoding='utf-8')

# Ensure multiprocessing-safe code execution
if __name__ == '__main__':
    process_papers_optimised(df_main,"DBLP_topics.csv")
# end




"""
def process_papers_only_syntatic(df):
    # Remove rows where "abstract" is null or has the value 'N/A'
    df_filtered = df[(df['abstracts'].notnull()) & (df['abstracts'] != 'N/A')& (df['abstracts'] != 'No abstracts available')]

    # Create the dictionary `paper`
    papers = {}
    for _, row in df_filtered.iterrows():
        doi = row['doi']
        papers[doi] = {
            "title": row['publication_title'],
            "abstract": row['abstracts']
        }

    start_time = time.time()

    cc = CSOClassifier(workers = 4, modules = "syntactic", enhancement = "all",
                        explanation = False, delete_outliers=True,
                        fast_classification = True, silent = False)
    result = cc.batch_run(papers)

    df_filtered['syntactic'] = None
    df_filtered['enhanced'] = None

    # Populate the correct row in df_filtered by matching DOI
    for doi, classification in result.items():
        if doi in df_filtered['doi'].values:

            enhanced = ', '.join(classification.get('enhanced', [])) 
            syntactic = ', '.join(classification.get('syntactic', [])) 

            df_filtered.loc[df_filtered['doi'] == doi, 'enhanced'] = enhanced
            df_filtered.loc[df_filtered['doi'] == doi, 'syntactic'] = syntactic

    # Total elapsed time
    total_time = time.time()
    print(f"Running the classifier on:\n{len(df_filtered.index)} number of rows took:\n{total_time - start_time:.2f} seconds")

    #df_filtered.to_csv('DBLP_a_star_venues_w_topics.csv', index=False)
    return df_filtered
"""

# Usage
# df_filtered = classify_papers_in_batches(df)