import pandas as pd

#### Clean CORE acronyms of 55 mislabel DBLP acronyms
'''
### Give ICORE csv propper headers
file=""
df = pd.read_csv(file, header=None)
header= ["id","title","acronym","source","rank","6","7","8","9"]
df.columns = header

#df

# region
df['acronym'] = df['acronym'].replace('ACMMM', 'ACM Multimedia')
df['acronym'] = df['acronym'].replace('EuroCrypt', 'EUROCRYPT')
df['acronym'] = df['acronym'].replace('MOBICOM', 'MobiCom')
df['acronym'] = df['acronym'].replace('SENSYS', 'SenSys')
df['acronym'] = df['acronym'].replace('SIGMOD', 'SIGMOD Conference')


# Add an alternative acronym to SIGMOD conference
sigmod_extra_acronym = df[df['acronym'] == 'SIGMOD Conference'].copy()
sigmod_extra_acronym['acronym'] = 'SIGMOD Conference Companion'

# Append the new row to the original df DataFrame
df = pd.concat([df, sigmod_extra_acronym], ignore_index=True)

df['acronym'] = df['acronym'].replace('SIGSPATIAL', 'SIGSPATIAL/GIS')  # GIS
df['acronym'] = df['acronym'].replace('IEEE VIS', 'VIS')
df['acronym'] = df['acronym'].replace('MOBIHOC', 'MobiHoc')
df['acronym'] = df['acronym'].replace('MSWIM', 'MSWiM')
df['acronym'] = df['acronym'].replace('ECML PKDD', 'PKDD')
df['acronym'] = df['acronym'].replace('Mobisys', 'MobiSys')
df['acronym'] = df['acronym'].replace('CaiSE', 'CAiSE')
df['acronym'] = df['acronym'].replace('FOSSACS', 'FoSSaCS')
df['acronym'] = df['acronym'].replace('IEEE CCNC', 'CCNC')
df['acronym'] = df['acronym'].replace('IEEE-IV', 'IV')
df['acronym'] = df['acronym'].replace('PKC', 'Public Key Cryptography')
df['acronym'] = df['acronym'].replace('EUROGP', 'EuroGP')
df['acronym'] = df['acronym'].replace('EuroPar', 'Euro-Par')
df['acronym'] = df['acronym'].replace('I3DG', 'I3D')  # SI3D
df['acronym'] = df['acronym'].replace('ACM_WiSec', 'WISEC')
df['acronym'] = df['acronym'].replace('SoCS', 'SOCS')
df['acronym'] = df['acronym'].replace('IEEE SSE', 'IEEE SCC')
df['acronym'] = df['acronym'].replace('SGP', 'Symposium on Geometry Processing')
df['acronym'] = df['acronym'].replace('Group', 'GROUP')


# Add an alternative acronym to SIGSPATIAL/GIS conference
SIGSPATIAL_extra_acronym = df[df['acronym'] == 'SIGSPATIAL/GIS'].copy()
SIGSPATIAL_extra_acronym['acronym'] = 'GIS'
# Append the new row to the original df DataFrame
df = pd.concat([df, SIGSPATIAL_extra_acronym], ignore_index=True)

# Add an alternative acronym to I3D conference
I3D_extra_acronym = df[df['acronym'] == 'I3D'].copy()
I3D_extra_acronym['acronym'] = 'SI3D'
# Append the new row to the original df DataFrame
df = pd.concat([df, I3D_extra_acronym], ignore_index=True)

df['acronym'] = df['acronym'].replace('SASHIMI', 'SASHIMI@MICCAI')
df['acronym'] = df['acronym'].replace('Tencon', 'TENCON')
df['acronym'] = df['acronym'].replace('COG', 'CoG')
df['acronym'] = df['acronym'].replace('Coordination', 'COORDINATION')
df['acronym'] = df['acronym'].replace('CSEET', 'CSEE&T')
df['acronym'] = df['acronym'].replace('DIGRA', 'DiGRA')
df['acronym'] = df['acronym'].replace('Euro VR', 'EuroXR')
df['acronym'] = df['acronym'].replace('EUSFLAT', 'EUSFLAT Conf.')
df['acronym'] = df['acronym'].replace('HASKELL', 'Haskell')
df['acronym'] = df['acronym'].replace('HOTCHIPS (HCS)', 'HCS')
df['acronym'] = df['acronym'].replace('ICCSA', 'ICCSA (1)')
df['acronym'] = df['acronym'].replace('IEEE CIFEr', 'CIFEr')
df['acronym'] = df['acronym'].replace('IEEE IS', 'IS')
df['acronym'] = df['acronym'].replace('IIWAS', 'iiWAS')
df['acronym'] = df['acronym'].replace('IWSM Mensura', 'IWSM/Mensura')
df['acronym'] = df['acronym'].replace('IJCRS (was RSCTC)', 'IJCRS')
df['acronym'] = df['acronym'].replace('OPENSYM', 'OpenSym')
df['acronym'] = df['acronym'].replace('Qshine', 'QSHINE')
df['acronym'] = df['acronym'].replace('ICSoft', 'ICSOFT')
df['acronym'] = df['acronym'].replace('Mobiquitous', 'MobiQuitous')
df['acronym'] = df['acronym'].replace('Broadnets', 'BROADNETS')
df['acronym'] = df['acronym'].replace('DIAGRAMS', 'Diagrams')
df['acronym'] = df['acronym'].replace('CICLING', 'CICLing')
df['acronym'] = df['acronym'].replace('DAFX', 'DAFx')
df['acronym'] = df['acronym'].replace('APWEB', 'APWeb')
df['acronym'] = df['acronym'].replace('MOMM', 'MoMM')
df['acronym'] = df['acronym'].replace('Algosensors', 'ALGOSENSORS')

# Add an alternative acronym to IJCRS conference
IJCRS_extra_acronym = df[df['acronym'] == 'IJCRS'].copy()
IJCRS_extra_acronym['acronym'] = 'RSCTC'
# Append the new row to the original df DataFrame
df = pd.concat([df, IJCRS_extra_acronym], ignore_index=True)
# endregion
df.to_csv(file)

'''

#### Add all core files into a df with all ranks for all venues where a area was found
#region

import pandas as pd

# Load CORE2023.csv and df_w_topics_#ofAuthors.csv
core_2023 = pd.read_csv('')
df_w_topics = pd.read_csv('')

# Drop duplicates from df_w_topics based on subset of 'venue_title' and 'venue_acronym'
df_w_topics.drop_duplicates(subset=['venue_title', 'venue_acronym'], inplace=True)

# Perform the merge and allow columns with the same name to get suffixes
df_merged = pd.merge(df_w_topics, core_2023, 
                     left_on=['venue_title', 'venue_acronym'], 
                     right_on=['title', 'acronym'], 
                     how='left', 
                     suffixes=('_x', '_y'))  # Handle naming conflicts by giving suffixes

# If both 'rank_x' and 'rank_y' exist, check if the values are the same
if 'rank_x' in df_merged.columns and 'rank_y' in df_merged.columns:
    if df_merged['rank_x'].equals(df_merged['rank_y']):
        # If they are equal, we can safely drop one of them
        df_merged.drop(columns=['rank_y'], inplace=True)  # Drop 'rank_y'
        df_merged.rename(columns={'rank_x': 'rank'}, inplace=True)  # Rename 'rank_x' back to 'rank'
    else:
        print("Warning: 'rank_x' and 'rank_y' have mismatched values. Doing a manual review.")
        print(df_merged[['title', 'acronym', 'rank_x', 'rank_y']])
else:
    # If only one exists (rank_x or rank_y), rename it to 'rank'
    df_merged.rename(columns={'rank_x': 'rank', 'rank_y': 'rank'}, inplace=True)

# Create df_new from the cleaned merged data
df_new = df_merged[['title', 'acronym', 'rank']]

# Add a new column with the source value (Since 'source' is uniform, it can be a constant).
# Assuming the 'source' value in CORE2023.csv is uniform - you're taking first instance.
source_value = core_2023['source'].iloc[0]  # Or replace with any known constant source
df_new[source_value] = df_new['rank']

# Drop the redundant 'rank' column, since its values are now in the 'source_value' column
df_new = df_new.drop(columns=['rank'])

# Function to add ranks from additional CORE csv files based on 'title' + 'acronym'
def add_ranks_from_core_file(file_path, df_new, source_column_name):
    # Read the CORE ranking CSV file
    core_df = pd.read_csv(file_path)
    
    # Merge df_new with the current core_df on both 'title' and 'acronym'
    df_new = pd.merge(
        df_new, 
        core_df[['title', 'acronym', 'rank']],  # Make sure the file has these columns
        on=['title', 'acronym'], 
        how='left', 
        suffixes=('', f'_{source_column_name}')
    )
    
    # Rename the resulting 'rank' column to the specific source name (e.g., the year)
    df_new.rename(columns={'rank': source_column_name}, inplace=True)
    
    return df_new

# List of file paths to the other core rank CSV files
core_files = [
    "abstract_API/CORE ranks/CORE2021.csv", 
    "abstract_API/CORE ranks/CORE2020.csv", 
    "abstract_API/CORE ranks/CORE2018.csv", 
    "abstract_API/CORE ranks/CORE2017.csv", 
    "abstract_API/CORE ranks/CORE2014.csv", 
    "abstract_API/CORE ranks/CORE2013.csv", 
    "abstract_API/CORE ranks/CORE2010.csv", 
    "abstract_API/CORE ranks/CORE2008.csv"
]

# Iterate over the CORE CSV files, adding a column to 'df_new' for each file
for file_path in core_files:
    # Extract the source year or identifier from the filename (e.g., CORE2021 from the path)
    source_year = file_path.split('/')[-1].split('.')[0]
    
    # Add ranks from the current core file to df_new using the updated function
    df_new = add_ranks_from_core_file(file_path, df_new, source_year)

df_new=df_new.drop_duplicates()

# Optional: Save the final df_new to a CSV file
df_new.to_csv('', index=True, header=True)

#endregion



### Make a df with a rank where all publications matches the CORE RANKING!
import pandas as pd

def filter_by_core_rank(df_topics, df_core, rank):
    # Merge the two dataframes on the respective columns acronym & venue_acronym
    merged_df = pd.merge(df_topics, df_core, left_on='venue_acronym', right_on='acronym', how='left')
    
    # Define the new fallback mapping for CORE ranking
    year_to_rank_column = {
        2023: 'CORE2023',
        2022: 'CORE2021',  # fallback to most recent
        2021: 'CORE2021',
        2020: 'CORE2020',
        2019: 'CORE2018',  # fallback
        2018: 'CORE2018',
        2017: 'CORE2017',
        2016: 'CORE2014',  # fallback
        2015: 'CORE2014',  # fallback
        2014: 'CORE2014',
        2013: 'CORE2013',
        2012: 'CORE2010',  # but specific logic for A* (fallback to 2008)
        2011: 'CORE2010',  # but specific logic for A* (fallback to 2008)
        2010: 'CORE2010',  # but specific logic for A* (fallback to 2008)
        2009: 'CORE2008',  # fallback
        2008: 'CORE2008'
    }
    
    def filter_by_year_and_rank(row, rank):
        # If the row year is in our defined range, check appropriate column
        if row['year'] in year_to_rank_column:
            core_rank_column = year_to_rank_column[row['year']]
            
            # Special case for years 2010, 2011, 2012 and rank "A*"
            if row['year'] in {2010, 2011, 2012} and rank == "A*":
                if pd.isna(row[core_rank_column]) or row[core_rank_column] != "A*":
                    # Fallback to CORE2008 if no A* in CORE2010 or others
                    return row['CORE2008'] == "A*"
            
            # General case: check if rank matches the specified rank
            return row[core_rank_column] == rank
        
        return False  # If the year is out of range, don't include that row

    # Apply filtering based on desired rank
    filtered_df = merged_df[merged_df.apply(filter_by_year_and_rank, axis=1, rank=rank)]
    
    return filtered_df

# Usage example:
# Load the CSV files (example file paths)
df_w_topics_Authors = pd.read_csv('')
core_df = pd.read_csv('')

# Filter the publications for the specified core rank 
result = filter_by_core_rank(df_w_topics_Authors, core_df, 'C')

# Save the filtered results
result.to_csv("", index=True)



### Which venues changes rank over time?

import pandas as pd

# Load the CSV into a pandas dataframe
df = pd.read_csv('')

# List of the CORE columns you are interested in
core_columns = ['CORE2023', 'CORE2021', 'CORE2020', 'CORE2018', 
                'CORE2017', 'CORE2014', 'CORE2013', 'CORE2010', 'CORE2008']

# Step 1: Create a copy of the dataframe for comparison purposes
df_comparison = df.copy()

# Step 2: For comparison purposes, treat A* and A as the same for CORE2010
# We'll replace 'A*' with 'A' in the 'CORE2010' column in the comparison dataframe
df_comparison['CORE2010'] = df_comparison['CORE2010'].replace({'A*': 'A'})

# Step 3: Now apply the logic to find rows where rankings changed or didn't change
# Compare based on the modified dataframe (df_comparison)

# Create a boolean mask where values across all CORE columns are not identical (changes found)
mask_changes = df_comparison[core_columns].nunique(axis=1) > 1

# Create the inverse mask where values across all CORE columns are identical (no change)
mask_no_changes = df_comparison[core_columns].nunique(axis=1) == 1

# Step 4: Use the original dataframe to output the desired result with unchanged values
df_with_changes = df[mask_changes]
df_no_changes = df[mask_no_changes]

# Optionally, save both DataFrames to CSV files
df_with_changes.to_csv('', index=False)
df_no_changes.to_csv('', index=False)

