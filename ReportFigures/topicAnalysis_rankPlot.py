import pandas as pd
import matplotlib.pyplot as plt
import time

# List of the first children of the root 'Computer Science'
superTopics = [
    "artificial intelligence", "bioinformatics", "computer aided design", 
    "computer hardware", "computer imaging and vision", "computer networks", 
    "computer programming", "computer security", "computer systems", 
    "data mining", "human computer interaction", "information retrieval", 
    "information technology", "internet", "operating systems", 
    "robotics", "software", "software engineering", "theoretical computer science"
]

def get_topic_colors(topics):
    cmap = plt.get_cmap('tab20')  # Using tab20 which contains 20 distinct colors
    colors = cmap.colors[:len(topics)]  # Limit to the number of distinct topics
    return dict(zip(topics, colors))

# Generate color mapping for superTopics
topic_to_color = get_topic_colors(superTopics)

# Function for processing each DataFrame and calculating fractional contributions
def calculate_fractions(df):
    print(f"Processing DataFrame with {len(df)} rows.")
    start_time = time.time()

    # Empty list to store the matched rows with fractions
    superTopic_fractions = []

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        try:
            # Check if 'enhanced' is not NaN (missing value) and not empty
            if pd.notna(row['enhanced']) and row['enhanced'].strip():
                # Split the 'enhanced' column list of topics into individual topics
                topics = row['enhanced'].split(', ')

                # Find the matched super topics
                matched_topics = [topic for topic in topics if topic.strip() in superTopics]

                # Calculate fractional count for each matched topic: 1 divided by the total length of matched superTopics
                if matched_topics:
                    fractional_value = 1 / len(matched_topics)
                    for topic in matched_topics:
                        superTopic_fractions.append({'year': row['year'], 'enhanced': topic.strip(), 'fraction': fractional_value})

        except Exception as e:
            print(f"An error occurred at row {index}: {e}")

    # Convert the list to a DataFrame
    fractional_df = pd.DataFrame(superTopic_fractions)

    # Calculate the sum of fractions for each super topic per year
    fraction_sum_per_year = fractional_df.groupby(['year', 'enhanced'])['fraction'].sum().reset_index()

    # Total elapsed time
    total_time = time.time()
    print(f"Processed {len(df)} rows in {total_time - start_time:.2f} seconds.")

    return fraction_sum_per_year

# Function to process a given dataframe
def process_df(df, sorted_total_fractions):
    # Filter df to include only the topics in sorted_total_fractions
    df = df[df['enhanced'].isin(sorted_total_fractions.index)]
    
    # Calculate the percentage for stacked plot
    df['total_per_year'] = df.groupby('year')['fraction'].transform('sum')
    df['percentage'] = (df['fraction'] / df['total_per_year']) * 100
    
    # Pivot the data for stacking and order columns by the sorted order
    df_pivot = df.pivot(index='year', columns='enhanced', values='percentage')
    df_pivot = df_pivot[sorted_total_fractions.index]  # Reorder columns
    return df_pivot

# List of CSV files
csv_files = ["A*_publications.csv", "A_publications.csv", "B_publications.csv", "C_publications.csv"]
titles = ["A* Conferences", "A Conferences", "B Conferences", "C Conferences"]

# Read all datasets first and store as DataFrames
dfs = [pd.read_csv(csv_file) for csv_file in csv_files]

# Process all datasets (including A* which is already done) using calculate_fractions
processed_dfs = [calculate_fractions(df) for df in dfs]

# Process the A* dataset first to get the ordering of topics
processed_A_star = processed_dfs[0]  # First df corresponds to A*_publications.csv

# Calculate total fraction contributions for sorting (based on A*)
total_fractions_A_star = processed_dfs[0].groupby('enhanced')['fraction'].sum().reset_index()

# Sort the topics by their total fraction in descending order
sorted_total_fractions = total_fractions_A_star.set_index('enhanced').sort_values(by='fraction', ascending=False)

# Bold years to mark explicitly, but exclude 2010 for A*
bold_years = [2008, 2013, 2014, 2017, 2018, 2020, 2021, 2023]
bold_years_all = [2008, 2010, 2013, 2014, 2017, 2018, 2020, 2021, 2023]  # For the other plots

# Now plot for each processed DataFrame in a 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)

axes = axes.flatten()  # Flatten the axes array to iterate over it

for i, (processed_df, ax, title) in enumerate(zip(processed_dfs, axes, titles)):
    # Process the dataframe, ensuring it uses the sorted order from A*_publications
    df_pivot = process_df(processed_df, sorted_total_fractions)

    # Prepare the colors for the stacked bar chart
    sorted_topics = df_pivot.columns
    plot_colors = [topic_to_color.get(t, 'gray') for t in sorted_topics]  # Use pre-assigned colors

    # Plot the stacked bar plot on the given axis
    df_pivot.plot(kind='bar', stacked=True, ax=ax, color=plot_colors, width=0.97)

    # Set titles and labels
    ax.set_title(title)

    # No x-axis label ('Year')
    ax.set_xlabel('')  # We're removing this and adding a single label at the bottom for the whole grid

    # Y-axis label will be added globally, not per axis
    ax.set_ylabel('')

    # X-tick formatting with bold years
    xticks = ax.get_xticks()
    xtick_labels = list(ax.get_xticklabels())

    for label in xtick_labels:
        year = int(label.get_text())
        # Check which list of bold years to use
        if i == 0:  # For A* plot, skip 2010
            if year in bold_years:
                label.set_fontweight('bold')  # Make the selected years bold
        else:  # For other plots, include 2010
            if year in bold_years_all:
                label.set_fontweight('bold')

    # Apply the new labels and rotate for readability
    ax.set_xticklabels(xtick_labels, rotation=45)

    # Remove individual legends from subplots
    ax.get_legend().remove()

# Create the consolidated legend shown once for the whole figure
handles, labels = axes[0].get_legend_handles_labels()  # Get labels from the first axes
def skip_and(label):
    return label.title().replace(" And ", " and ")
if labels is not None:
    labels = [skip_and(label) for label in labels]
fig.legend(handles[::-1], labels[::-1], loc='center right', title='Areas')

# Add a single Y-axis label for the whole grid, placed on the left
fig.text(0.04, 0.5, 'Percentage (%)', va='center', rotation='vertical', fontsize=12)

# Set Y-axis limit to ensure it ranges from 0% to 100% for each axis
for ax in axes:
    ax.set_ylim(0, 100)  # Ensure the Y-axis goes from 0% to 100%

# Add a single X-axis label for the whole grid, placed below
fig.text(0.47, 0.02, 'Years\n(CORE ranking years in$\\mathbf{\ bold}$)', ha='center', fontsize=12)

# Add a title to the grid using fig.text, positioned slightly above the grid
fig.text(0.47, 0.945, "Publication Areas of Ranks Over Time", ha='center', fontsize=16, weight='bold')

# Adjust the layout to ensure everything fits including the title, axis labels, and legend
plt.tight_layout(rect=[0.05, 0.05, 0.78, 0.93])  # Adjust the top margin slightly

# Show the final plot
plt.show()

# Save the plot
plt.savefig('')
plt.savefig('')



###
# find avg percentage
###
df_Astar = process_df(processed_dfs[0],sorted_total_fractions)
df_A = process_df(processed_dfs[1],sorted_total_fractions)
df_B = process_df(processed_dfs[2],sorted_total_fractions)
df_C = process_df(processed_dfs[3],sorted_total_fractions)

def get_average_percentage(df_Astar, df_A, df_B, df_C, area, year1, year2):
    # Filter the data by the year interval (inclusive)
    df_Astar_filtered = df_Astar.loc[year1:year2, area]
    df_A_filtered = df_A.loc[year1:year2, area]
    df_B_filtered = df_B.loc[year1:year2, area]
    df_C_filtered = df_C.loc[year1:year2, area]
    
    # Calculate the average percentage for each DataFrame
    avg_Astar = df_Astar_filtered.mean()
    avg_A = df_A_filtered.mean()
    avg_B = df_B_filtered.mean()
    avg_C = df_C_filtered.mean()
    
    # Return the results as a dictionary or a DataFrame
    return {
        'df_Astar': avg_Astar,
        'df_A': avg_A,
        'df_B': avg_B,
        'df_C': avg_C
    }

# Example usage:
averages = get_average_percentage(df_Astar, df_A, df_B, df_C, 'computer hardware', 2008, 2023)
print(averages)
print("2016->2013")