import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

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

# Inset publication dataset
df_main = pd.read_csv("df_w_topics_#ofAuthors.csv")

### ------------------------------------------------------------------------------
# Plot functions
def LinePlot(df,year1=None,year2=None,topConf=None,title=None):
    year1 = 1967 if year1 is None else year1
    year2 = 2023 if year2 is None else year2
    topConf = 19 if topConf is None else topConf
    # Filter data for the year interval
    df_year_interval = df[
        (df['year'] >= year1) & 
        (df['year'] <= year2)
    ]

    # Calculate total fraction contributions for sorting
    total_fractions = df_year_interval.groupby('enhanced')['fraction'].sum().reset_index()

    # Get the top five topics by total fraction
    top_five_topics = total_fractions.nlargest(topConf, 'fraction')['enhanced']  

    # Sort these topics based on their total contributions
    sorted_top_five_topics = top_five_topics.sort_values(
        key=lambda topic: total_fractions.set_index('enhanced').loc[topic]['fraction'],
        ascending=False
    )

    # Filter df_year_interval to include only the top five topics
    df_year_interval = df_year_interval[df_year_interval['enhanced'].isin(sorted_top_five_topics)]

    # Set up the plot
    plt.figure(figsize=(16, 8))

    # Plotting each topic in the sorted order for consistent legend ordering
    for topic in sorted_top_five_topics:
        subset = df_year_interval[df_year_interval['enhanced'] == topic]
        sns.lineplot(data=subset, x='year', y='fraction', label=topic, marker='o')

    # Set plot title and labels
    plt.title(f'Count of Super Topics Over Years ({year1}-{year2})\n{title}')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.grid(True)

    # Display the plot
    plt.show()

def StackedBarPlot(df, title=None):
    # Default parameters

    # Calculate total fraction contributions for sorting
    total_fractions = df.groupby('enhanced')['fraction'].sum().reset_index()

    # Sort the topics by their total fraction in ascending order (smallest first)
    sorted_total_fractions = total_fractions.set_index('enhanced').sort_values(by='fraction', ascending=False)

    # Filter df to include only the top topics
    df = df[df['enhanced'].isin(sorted_total_fractions.index)]

    # Calculate the percentage for stacked plot
    df['total_per_year'] = df.groupby('year')['fraction'].transform('sum')
    df['percentage'] = (df['fraction'] / df['total_per_year']) * 100

    # Pivot the data for stacking and order columns by sorted (smallest-first) topics
    df_pivot = df.pivot(index='year', columns='enhanced', values='percentage')
    df_pivot = df_pivot[sorted_total_fractions.index]  # Reorder columns to match sorted order

    # Step 2: Use specific colors for each plotted topic based on the predefined color dictionary
    sorted_topics = df_pivot.columns
    plot_colors = [topic_to_color.get(t, 'gray') for t in sorted_topics]  # Apply assigned colors, gray if missing

    # Plotting the stacked bar plot
    ax = df_pivot.plot(kind='bar', stacked=True, figsize=(16, 8), color=plot_colors, width=0.99)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    # Add title and labels
    plt.title(f'Distribution of Areas Over Time\n{title}')
    plt.xlabel('Year of CORE ranking')
    plt.ylabel('Percentage (%)')
    plt.grid(True, axis='y')

    # Show the legend in the same sorted order (smallest first)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], title='Areas', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Display the plot
    plt.show()

#Outdated
'''

def StackedBarPlotInterval(df, year1=1967, year2=2023, topConf=19, title=None, interval=2, save=False):

    # Filter data for the year interval
    df_year_interval = df[(df['year'] >= year1) & (df['year'] <= year2)]

    # Create a new column for grouping years by a specified interval
    df_year_interval['year_interval'] = (df_year_interval['year'] - year1) // interval * interval + year1
    df_year_interval['year_interval'] = df_year_interval['year_interval'].astype(int)

    # Create a label for each group like '1999/2000'
    df_year_interval['year_label'] = df_year_interval.apply(
        lambda row: f"{row['year_interval']}-{min(row['year_interval'] + interval - 1, year2)}", axis=1
    )

    # Calculate total fraction contributions for sorting
    total_fractions = df_year_interval.groupby('enhanced')['fraction'].sum().reset_index()

    # Sort topics by total fraction
    sorted_topics = total_fractions.nlargest(topConf, 'fraction').sort_values(by='fraction', ascending=False)['enhanced']

    # Filter to include only the top topics
    df_year_interval = df_year_interval[df_year_interval['enhanced'].isin(sorted_topics)]

    # Group by year_label and enhanced to calculate percentages
    df_interval_grouped = df_year_interval.groupby(['year_label', 'enhanced'])['fraction'].sum().reset_index()
    df_interval_grouped['total_per_interval'] = df_interval_grouped.groupby('year_label')['fraction'].transform('sum')
    df_interval_grouped['percentage'] = (df_interval_grouped['fraction'] / df_interval_grouped['total_per_interval']) * 100

    # Pivot for plotting
    df_pivot = df_interval_grouped.pivot(index='year_label', columns='enhanced', values='percentage')
    df_pivot = df_pivot[sorted_topics]  # Reorder columns by sorted topics

    # Assign colors to topics
    plot_colors = [topic_to_color.get(topic, 'gray') for topic in df_pivot.columns]

    # Plotting the stacked bar plot
    ax = df_pivot.plot(kind='bar', stacked=True, figsize=(16, 8), color=plot_colors, width=0.98)

    ax.set_ylim(0, 100)

    # Add title and labels
    plt.title(f'Distribution of CS Areas Over Years ({year1}-{year2}) in {interval}-Year Intervals\n{title}')
    plt.xlabel(f'{interval}-Year Intervals')
    plt.ylabel('Percentage (%)')
    plt.grid(True, axis='y')

    # Legend customization
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], title='Areas', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    if save==False:
        plt.show()
    else:
        plt.savefig(f'all_ranks_[{interval}].pdf')
        plt.savefig(f'all_ranks_[{interval}].png')


'''

def makeRelative(df, year1=1967, year2=2023, topConf=19, interval=2):
    # Filter data for the year interval
    df_year_interval = df[(df['year'] >= year1) & (df['year'] <= year2)]

    # Create a new column for grouping years by a specified interval
    df_year_interval['year_interval'] = (df_year_interval['year'] - year1) // interval * interval + year1
    df_year_interval['year_interval'] = df_year_interval['year_interval'].astype(int)

    # Create a label for each group like '1999/2000'
    df_year_interval['year_label'] = df_year_interval.apply(
        lambda row: f"{row['year_interval']}-{min(row['year_interval'] + interval - 1, year2)}", axis=1
    )

    # Calculate total fraction contributions for sorting
    total_fractions = df_year_interval.groupby('enhanced')['fraction'].sum().reset_index()

    # Sort topics by total fraction
    sorted_topics = total_fractions.nlargest(topConf, 'fraction').sort_values(by='fraction', ascending=False)['enhanced']

    # Filter to include only the top topics
    df_year_interval = df_year_interval[df_year_interval['enhanced'].isin(sorted_topics)]

    # Group by year_label and enhanced to calculate percentages
    df_interval_grouped = df_year_interval.groupby(['year_label', 'enhanced'])['fraction'].sum().reset_index()
    df_interval_grouped['total_per_interval'] = df_interval_grouped.groupby('year_label')['fraction'].transform('sum')
    df_interval_grouped['percentage'] = (df_interval_grouped['fraction'] / df_interval_grouped['total_per_interval']) * 100

    return df_interval_grouped, sorted_topics

def StackedBarPlotInterval(df, year1=1967, year2=2023, topConf=19, title=None, interval=2, save=False):
    df_interval_grouped, sorted_topics= makeRelative(df,year1,year2,topConf,interval)

    # Pivot for plotting
    df_pivot = df_interval_grouped.pivot(index='year_label', columns='enhanced', values='percentage')
    df_pivot = df_pivot[sorted_topics]  # Reorder columns by sorted topics

    # Assign colors to topics
    plot_colors = [topic_to_color.get(topic, 'gray') for topic in df_pivot.columns]

    # Plotting the stacked bar plot
    ax = df_pivot.plot(kind='bar', stacked=True, figsize=(16, 8), color=plot_colors, width=0.98)

    ax.set_ylim(0, 100)
    xtick_labels = list(ax.get_xticklabels())
    ax.set_xticklabels(xtick_labels,rotation=45)

    # Add title and labels
    plt.title(f'Distribution of CS Areas Over Years ({year1}-{year2}) in {interval}-Year Intervals\n{title}', ha='center', fontsize=16, weight='bold')
    plt.xlabel(f'{interval}-Year Intervals',labelpad=15)
    plt.ylabel('Percentage (%)')
    plt.grid(True, axis='y')

    # Legend customization
    handles, labels = ax.get_legend_handles_labels()
    def skip_and(label):
        return label.title().replace(" And ", " and ")
    
    if labels is not None:
        labels = [skip_and(label) for label in labels]
    ax.legend(handles[::-1], labels[::-1], title='Areas', bbox_to_anchor=(1.00, 1), loc='upper left')

    plt.tight_layout()
    if save==False:
        plt.show()
    else:
        plt.savefig(f'all_ranks_[{interval}].pdf')
        plt.savefig(f'all_ranks_[{interval}].png')

def StackedBarPlotForTopicsSameProportions(df, topics, year1=1967, year2=2023, interval=2, title=None, save=False):
    # Ensure topics list is valid
    topics = [topic.strip().lower() for topic in topics]

    # Calculate proportions for all topics
    df_interval_grouped, sorted_topics = makeRelative(df, year1, year2, len(df['enhanced'].unique()), interval)

    # Standardize topic names for matching
    df_interval_grouped['enhanced'] = df_interval_grouped['enhanced'].str.strip().str.lower()

    # Filter DataFrame to include only the specified topics
    df_filtered = df_interval_grouped[df_interval_grouped['enhanced'].isin(topics)]

    if df_filtered.empty:
        raise ValueError(f"None of the specified topics {topics} are present in the data after filtering.")

    # Pivot for plotting
    df_pivot = df_filtered.pivot(index='year_label', columns='enhanced', values='percentage')

    # Ensure topics exist in columns after pivot
    missing_topics = [topic for topic in topics if topic not in df_pivot.columns]
    if missing_topics:
        raise ValueError(f"The following topics are missing from the pivot table: {missing_topics}")

    df_pivot = df_pivot[topics]  # Reorder columns to match selected topics

    # Assign colors to topics
    plot_colors = [topic_to_color.get(topic, 'gray') for topic in df_pivot.columns]

    # Plotting the stacked bar plot
    ax = df_pivot.plot(kind='bar', stacked=True, figsize=(16, 8), color=plot_colors, width=0.98)

    ax.set_ylim(0, 100)
    ax.set_xticklabels(rotation=45)

    # Add title and labels
    title_suffix = f" ({', '.join(topics)})" if title is None else title
    plt.title(f'Distribution of Selected CS Areas Over Years ({year1}-{year2}) in {interval}-Year Intervals\n{title_suffix}')
    plt.xlabel(f'{interval}-Year Intervals',labelpad=20)
    plt.ylabel('Percentage (%)')
    plt.grid(True, axis='y')

    ax.yaxis.set_label_coords(-0.03, 0.11)
    # Legend customization
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], title='Areas', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    if save:
        filename = f"selected_topics_same_proportions_{'_'.join(topics)}_[{interval}].pdf"
        plt.savefig(filename.replace('.pdf', '.pdf'))
        plt.savefig(filename.replace('.pdf', '.png'))
    else:
        plt.show()

#Outdated
'''
def oldHeatmapInteractions(df, labels=superTopics,title='Heatmap of Topic Interactions',save=False):
    def skip_and(label):
        return label.title().replace(" And ", " and ")
    
    if labels is not None:
        labels = [skip_and(label) for label in labels]

    def custom_format(value):
        if value == 0:
            return ""  
        formatted = f"{value:.2g}"  # Two significant digits
        if len(formatted)==1:
            formatted = f"{int(round(float(formatted))):.1f}"  # Ensure integer has .0
        return formatted

    plt.figure(figsize=(12, 9))
    ax = sns.heatmap(df,
                xticklabels=labels,
                yticklabels=labels,
                cmap='Blues', 
                linewidth=1,
                #annot=np.array([[custom_format(val) for val in row] for row in df]),#annot=True, 
                #fmt='',#'.2g',
                cbar_kws={'format': '%.0f%%', 'pad': 0.02}
                )
    ax.figure.axes[-1].set_ylabel('Interaction Percentage (%)', labelpad=15)
    ax.set(xlabel="", ylabel="")
    ax.xaxis.tick_top()
    plt.title(title, pad=20)
    plt.xticks(rotation=90)  
    plt.tight_layout()
    if save==False:
        plt.show()
    else:
        plt.savefig(f'relative.png')
        plt.savefig(f'relative.pdf')
'''

def HeatmapInteractions(df, labels=None, title='Heatmap of Topic Collaborations', save=False):
    import numpy as np
    
    def skip_and(label):
        return label.title().replace(" And ", " and ")
    
    if labels is not None:
        labels = [skip_and(label) for label in labels]

    def custom_format(value):
        if value == 0:
            return ""  
        formatted = f"{value:.2g}"  # Two significant digits
        if len(formatted) == 1:
            formatted = f"{int(round(float(formatted))):.1f}"  # Ensure integer has .0
        return formatted

    plt.figure(figsize=(12, 10))  
    ax = sns.heatmap(
        df,
        xticklabels=labels,
        yticklabels=labels,
        cmap='Blues',
        linewidth=1,
        cbar_kws={
            'format': '%.0f%%',
            'orientation': 'horizontal',
            'pad': 0.01,  # Reduce distance from plot
            'aspect': 40  # Control thickness
        },
        vmin=np.max(df),  # Reverse the colorbar
        vmax=np.min(df)
    )
    
    # Adjust the colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.xaxis.set_label_position('bottom')
    cbar.ax.set_xlabel('Collaboration Percentage (%)', labelpad=10)  # Adjust label padding
    cbar.ax.invert_xaxis()  # Invert the range to go from highest to lowest

    ax.set(xlabel="", ylabel="")
    ax.xaxis.tick_top()
    plt.title(title, pad=20, ha='center', fontsize=16, weight='bold')
    plt.xticks(rotation=90)  
    plt.tight_layout()
    
    if not save:
        plt.show()
    else:
        plt.savefig(f'relative.png')
        plt.savefig(f'relative.pdf')

def PieChart(df, year1=None, year2=None, topConf=None, title=None):
    year1 = 1967 if year1 is None else year1
    year2 = 2023 if year2 is None else year2
    topConf = 19 if topConf is None else topConf
    
    # Filter data for the year interval
    df_year_interval = df[
        (df['year'] >= year1) & 
        (df['year'] <= year2)
    ]
    
    # Calculate total fraction contributions for sorting and grouping
    total_fractions = df_year_interval.groupby('enhanced')['fraction'].sum().reset_index()

    # Get the top topics by total fraction (up to `topConf`)
    top_topics = total_fractions.nlargest(topConf, 'fraction')['enhanced']  
    
    # Sort these topics based on their total contributions for pie chart order
    sorted_top_topics = top_topics.sort_values(
        key=lambda topic: total_fractions.set_index('enhanced').loc[topic]['fraction'],
        ascending=False
    )

    # Filter df_year_interval to include only the top topics
    df_top_topics = df_year_interval[df_year_interval['enhanced'].isin(sorted_top_topics)]

    # Re-calculate total fraction contributions with only top topics
    top_topic_fractions = df_top_topics.groupby('enhanced')['fraction'].sum().reset_index()

    # Sort the top topic fractions for pie chart consistency
    top_topic_fractions = top_topic_fractions.set_index('enhanced').loc[sorted_top_topics].reset_index()

    # Set up the Pie Chart
    plt.figure(figsize=(10, 7))

    # Plotting the pie chart
    plt.pie(
        top_topic_fractions['fraction'], 
        labels=top_topic_fractions['enhanced'], 
        autopct='%1.1f%%',  # Show percentage on the pie slices
        startangle=140,      # Start angle for the slices
        colors=plt.cm.Paired.colors,  # Assign colors to slices
        wedgeprops={'edgecolor': 'black'}  # Add some separation between the slices
    )

    # Set the title for the plot
    plt.title(f'Distribution of Super Topics ({year1}-{year2})\n{title}')

    # Ensure the pie is a circle and not an ellipse
    plt.axis('equal')

    # Show the pie chart
    plt.show()


### ------------------------------------------------------------------------------
# Give each super topic a value of the total amount of super topics for each publicaiton e.g.
# Publication: ("artificial intelligence": 1/3), ("bioinformatics": 1/3), ("computer aided design": 1/3)
# This is then grouped pr year and super topic where the fractions are summed together

# Executable code in region below:
# region
### ============================================================================== 
# Load the Csv file
df = df_main#.sample(n=100,random_state=1)

#df = df[(df['rank'] == "C")]

start_time = time.time()

# Empty list to store the matched rows with fractions, blueprint for final dataframe
superTopic_fractions = []

# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    try:
        # Check if 'enhanced' is not NaN (missing value) and not empty
        if pd.notna(row['enhanced']) and row['enhanced'].strip():
            # Split the 'enhanced' column list of topics into into individual topics
            topics = row['enhanced'].split(', ')
        
        # Find the matched super topics
        matched_topics = [topic for topic in topics if topic.strip() in superTopics]

        # Calculate fractional count for each matched topic: 1 divided the total length of superTopics of the publication
        if matched_topics:
            fractional_value = 1 / len(matched_topics)
            for topic in matched_topics:
                superTopic_fractions.append({'year': row['year'], 'enhanced': topic.strip(), 'fraction': fractional_value})
    except Exception as e:
        print(e)
        break

# Convert the list to a DataFrame
fractional_df = pd.DataFrame(superTopic_fractions)

fractional_df.columns

# Calculate the sum of fractions for each super topic per year
fraction_sum_per_year = fractional_df.groupby(['year', 'enhanced'])['fraction'].sum().reset_index()

# Total elapsed time
total_time = time.time()
print(f"Running on a file with:\n{len(df.index)} number of rows took:\n{total_time - start_time:.2f} seconds")
### ==============================================================================
# endregion

LinePlot(fraction_sum_per_year,None,None,None,"1/len of superTopics pr publication")

PieChart(fraction_sum_per_year,2000,None,None,"1/len of superTopics pr publication")
PieChart(fraction_sum_per_year,None,2000,10,"1/len of superTopics pr publication")

StackedBarPlot(fraction_sum_per_year, "A* conferences")
StackedBarPlotInterval(fraction_sum_per_year, year1=1979, title="", interval=2, save=True)

StackedBarPlotForTopicsSameProportions(fraction_sum_per_year, topics=["computer programming"], year1=1979, interval=2, save=True)

#find the highst and lowest percentage
df_rel,temp = makeRelative(fraction_sum_per_year,year1=1979, year2=2023, topConf=19, interval=2)

def get_highlow_percentages(df, enhanced, highlowint):
    # Filter the DataFrame for the specified 'enhanced' field
    filtered_df = df[df['enhanced'] == enhanced]

    filtered_df = filtered_df.drop('fraction', axis=1)
    filtered_df = filtered_df.drop('total_per_interval', axis=1)
    filtered_df = filtered_df.reindex(columns=['enhanced', 'year_label', 'percentage'])  


    # Sort by percentage for highest and lowest
    highest = filtered_df.nlargest(highlowint, 'percentage').sort_values('year_label', ascending=False)
    lowest = filtered_df.nsmallest(highlowint, 'percentage').sort_values('year_label', ascending=False)

    # Combine the results
    result = pd.concat([highest, lowest]).sort_values('percentage', ascending=False)

    return highest,lowest

highest,lowest = get_highlow_percentages(df_rel, 'artificial intelligence', 5)
print(highest)
print(lowest)


### ------------------------------------------------------------------------------
# Heat maps
#
# 

# absolute Heat map 
# region
# Step 3: Initialize an interaction matrix (n x n)

n = len(superTopics)
interaction_matrix = np.zeros((n, n), dtype=int)

# Step 4: Process each row in the 'enhanced' column
for topics in df_main['enhanced'].dropna():  # Drop NaN values
    # Clean and split the topics
    topic_list = [topic.strip().lower() for topic in topics.split(',')]
    
    # Only consider topics present in superTopics
    filtered_topics = [topic for topic in topic_list if topic in [st.lower() for st in superTopics]]
    
    # Find all combinations of pairs and update the interaction matrix
    for topic1, topic2 in combinations(filtered_topics, 2):
        i, j = superTopics.index(topic1.lower()), superTopics.index(topic2.lower())
        interaction_matrix[i, j] += 1
        interaction_matrix[j, i] += 1  # Symmetric matrix

# Step 1: Compute the sum of interactions for each topic
topic_sums = interaction_matrix.sum(axis=1)  # Sum across rows (or axis 0 will work too since it's symmetric)

# Step 2: Get the sorted indices based on the sums in descending order
sorted_indices = np.argsort(topic_sums)[::-1]  # Sort in descending order

# Step 3: Reorder interaction_matrix and superTopics based on the sorted indices
sorted_matrix = interaction_matrix[sorted_indices, :][:, sorted_indices]  # Reorder rows and columns
sorted_superTopics = [superTopics[i] for i in sorted_indices]  # Reorder topic names

HeatmapInteractions(sorted_matrix, labels=sorted_superTopics)
# endregion

# relative Heat map
# region
# Assume interaction_matrix already contains counts from earlier steps
n = len(superTopics)
interaction_matrix = np.zeros((n, n), dtype=int)

# Step 1: Process each row in the 'enhanced' column
for topics in df_main['enhanced'].dropna():  # Drop NaN values
    # Clean and split the topics
    topic_list = [topic.strip().lower() for topic in topics.split(',')]
    
    # Only consider topics present in superTopics
    filtered_topics = [topic for topic in topic_list if topic in [st.lower() for st in superTopics]]
    
    # Find all combinations of pairs and update the interaction matrix
    for topic1, topic2 in combinations(filtered_topics, 2):
        i, j = superTopics.index(topic1.lower()), superTopics.index(topic2.lower())
        interaction_matrix[i, j] += 1
        interaction_matrix[j, i] += 1  # Symmetric matrix

# Step 2: Normalize interaction_matrix by row-wise sums to create a row normalized matrix
row_sums = interaction_matrix.sum(axis=1, keepdims=True)  # Calculate the sum of each row

# Avoid division by zero for rows that have no interactions
row_sums[row_sums == 0] = 1  

# Normalize each element in the row by the sum of its respective row (making sure each row sums to 100%)
normalized_interaction_matrix = (interaction_matrix / row_sums) * 100  # Scale to percentage

# Step 3: Calculate interactivity (row sums) and sort the rows and columns
total_interactivity = interaction_matrix.sum(axis=1)  # Total interactivity per row

# Get the sorted indices based on total interactivity in descending order
sorted_indices = np.argsort(total_interactivity)[::-1]

# Step 4: Reorder both the normalized matrix and the superTopics list according to sorted indices
sorted_normalized_matrix = normalized_interaction_matrix[sorted_indices, :][:, sorted_indices]
sorted_superTopics = [superTopics[i] for i in sorted_indices]

HeatmapInteractions(sorted_normalized_matrix, labels=sorted_superTopics, 
                    title="Area Collaborations\n(Row-Normalized Percentage)",
                    save=False)
# endregion


### ------------------------------------------------------------------------------
# Grouped pr year and pr numberOfCreators where the fractions count of each superTopic are summed together
# region
def find_fractional_topics(row):
    if pd.notna(row['enhanced']) and row['enhanced'].strip():
        # Split the 'enhanced' string into a list of topics
        topics = row['enhanced'].split(', ')
        # Find matched super topics
        matched_topics = [topic.strip() for topic in topics if topic.strip() in superTopics]
        
        if matched_topics:
            # Calculate the fractional count based on the number of matched topics
            fractional_value = 1 / len(matched_topics)
            # Return a list of dictionaries with year, numberOfAuthors, topic, and fractional value
            return [{'year': row['year'], 'numberOfCreators': row['numberOfCreators'], 
                     'enhanced': topic, 'fraction': fractional_value} for topic in matched_topics]
    return []

# Apply the function to each row and explode the result
all_fractions = df_main.apply(find_fractional_topics, axis=1).explode()

# Filter out empty rows
all_fractions = all_fractions[all_fractions.notna()]

# Convert the result into a DataFrame
fractional_df = pd.DataFrame(all_fractions.tolist())

# Group by year, numberOfAuthors, and topic, and sum the fractional counts
fraction_sum_per_year_authors = fractional_df.groupby(['year', 'numberOfCreators', 'enhanced'])['fraction'].sum().reset_index()
# endregion

# Relatiev representation year and pr numberOfCreators
#region
filtered=False
df_num = df_main[
    (df_main['year'] >= 1979) & 
    (df_main['year'] <= 2023)
]
# with interval
# Define bins for 'numberOfCreators'
bins = [1, 2, 3, 5, 9, 17, 256]  # Upper bounds of the bins
labels = ["1", "2", "3-4", "5-8", "9-16", "17+"]

# Step 1: Add new 'interval' column based on the bins for 'numberOfCreators'
df_num['interval'] = pd.cut(df_num['numberOfCreators'], bins=bins, labels=labels, right=False)

# Step 2: Create a new '2_year_range' column to group years into 2-year periods
df_num['2_year_range'] = df_num['year'].apply(lambda x: f"{x}" if x == 2023 else (f"{x}-{x+1}" if x % 2 == 1 else f"{x-1}-{x}"))

###### OPTIONAL Step 2.1: only include publications from a specific area
filtered=True
area='artificial intelligence'
df_filtered = df_num[df_num['enhanced'].str.contains(area, case=False, na=False)]

######

# Step 3: Group by the new '2_year_range' and 'interval', then sum the number of creators
if filtered:
    df_grouped = df_filtered.groupby(['2_year_range', 'interval'])['numberOfCreators'].sum().reset_index()
else:    
    df_grouped = df_num.groupby(['2_year_range', 'interval'])['numberOfCreators'].sum().reset_index()

# Step 4: Calculate the total number of creators per 2-year period
df_grouped['total_per_year'] = df_grouped.groupby('2_year_range')['numberOfCreators'].transform('sum')

# Step 5: Calculate the percentage of each 'interval' relative to the total per 2-year period
df_grouped['percentage'] = (df_grouped['numberOfCreators'] / df_grouped['total_per_year']) * 100

# Step 6: Pivot the data to create a format suitable for stacked bar plotting
df_pivot = df_grouped.pivot(index='2_year_range', columns='interval', values='percentage').fillna(0)

# Step 7: Plot the stacked bar chart
ax = df_pivot.plot(kind='bar', stacked=True, figsize=(13, 6), cmap='tab20c', width=0.95)

# Set Y-axis limit to ensure it ranges from 0% to 100% axis
ax.set_ylim(0, 100)  # Ensure the Y-axis goes from 0% to 100%

# Add title and labels
if filtered:
    plt.title(f'Distribution of Number of Authors Over 2-Year Ranges\n for {area}')
else:
    plt.title('Distribution of Number of Authors Over 2-Year Ranges')
plt.xlabel('2-Year Range')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=45)  
plt.grid(True, axis='y')

# Show the legend outside the plot
plt.legend(title='Number of Authors\n(Intervals)', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

#plt.savefig('all_ranks_[2].pdf')

# Step 8: Display the plot
plt.show()


# Filter rows for the year 2020
df_2020 = df_main[df_main['year'] == 2023]

# Calculate the average number of creators
average_creators_2020 = df_2020['numberOfCreators'].mean()

print("Average number of creators in 2023:", average_creators_2020)


# Step 1: Calculate average number of authors per year
avg_authors_per_year = df_main.groupby('year')['numberOfCreators'].mean().reset_index()

# Step 2: Perform a cubic polynomial fit
years = avg_authors_per_year['year']
avg_authors = avg_authors_per_year['numberOfCreators']

# Fit a cubic polynomial
coefficients = np.polyfit(years, avg_authors, 3)
cubic_function = np.poly1d(coefficients)

# Step 3: Generate predictions for a range of years
years_range = np.linspace(years.min(), years.max(), 100)  # Fine-grained range for smooth curve
predicted_authors = cubic_function(years_range)

# Step 4: Plot the data and cubic approximation
plt.scatter(years, avg_authors, color='red', label='Average Authors (Data)')
plt.plot(years_range, predicted_authors, color='blue', label='Cubic Approximation')
plt.xlabel('Year')
plt.ylabel('Average Number of Authors')
plt.title('Cubic Approximation of Average Number of Authors Per Paper')
plt.legend()
plt.grid()
plt.show()




# without interval
'''
# Define the intervals (buckets)
bins = [1, 2, 3, 5, 9, 17, 33, 65, 129, 256]  # Upper bounds of the bins
labels = ["1", "2", "3-4", "5-8", "9-16", "17-32", "33-64", "65-128", "129-256"]

# Step 1: Add a new column 'interval' that categorizes 'numberOfCreators' into the defined bins
df_num['interval'] = pd.cut(df_num['numberOfCreators'], bins=bins, labels=labels, right=False)

# Step 2: Group by 'year' and 'interval', then sum the number of creators
df_grouped = df_num.groupby(['year', 'interval'])['numberOfCreators'].sum().reset_index()

# Step 3: Calculate the total number of creators per year
df_grouped['total_per_year'] = df_grouped.groupby('year')['numberOfCreators'].transform('sum')

# Step 4: Calculate the percentage of each interval relative to the total per year
df_grouped['percentage'] = (df_grouped['numberOfCreators'] / df_grouped['total_per_year']) * 100

# Step 5: Pivot the data to get a format suitable for stacked bar plotting
df_pivot = df_grouped.pivot(index='year', columns='interval', values='percentage').fillna(0)

# Step 6: Plot the stacked bar chart
ax = df_pivot.plot(kind='bar', stacked=True, figsize=(16, 8), cmap='tab20', width=0.9)

# Add title and labels
plt.title('Relative Distribution of Number of Creators Over the Years')
plt.xlabel('Year')
plt.ylabel('Percentage (%)')
plt.grid(True, axis='y')

# Show the legend
plt.legend(title='Number of Creators (Intervals)', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Display the plot
plt.show()
'''
#endregion

# Relatiev representation year and pr numberOfCreators for a specfic area
def plot_author_distribution(df, area=None):
    # Step 1: Filter the dataframe by year
    df_filtered = df[(df['year'] >= 1979) & (df['year'] <= 2023)].copy()
    
    # Step 2: Define bins for 'numberOfCreators' and add 'interval' column
    bins = [1, 2, 3, 5, 9, 17, 256]
    labels = ["1", "2", "3-4", "5-8", "9-16", "17+"]
    df_filtered['interval'] = pd.cut(df_filtered['numberOfCreators'], bins=bins, labels=labels, right=False)
    
    # Step 3: Add '2_year_range' column
    df_filtered['2_year_range'] = df_filtered['year'].apply(
        lambda x: f"{x}" if x == 2023 else (f"{x}-{x+1}" if x % 2 == 1 else f"{x-1}-{x}")
    )
    
    # Step 4: Filter by area if specified
    if area:
        df_filtered = df_filtered[df_filtered['enhanced'].str.contains(area, case=False, na=False)]
    
    # Step 5: Group by '2_year_range' and 'interval', and sum 'numberOfCreators'
    df_grouped = df_filtered.groupby(['2_year_range', 'interval'])['numberOfCreators'].sum().reset_index()
    
    # Step 6: Calculate total per year and percentage
    df_grouped['total_per_year'] = df_grouped.groupby('2_year_range')['numberOfCreators'].transform('sum')
    df_grouped['percentage'] = (df_grouped['numberOfCreators'] / df_grouped['total_per_year']) * 100
    
    # Step 7: Pivot data for plotting
    df_pivot = df_grouped.pivot(index='2_year_range', columns='interval', values='percentage').fillna(0)
    
    # Step 8: Plot the stacked bar chart
    ax = df_pivot.plot(kind='bar', stacked=True, figsize=(13, 6), cmap='tab20c', width=0.95)
    ax.set_ylim(0, 100)
    
    # Add title and labels
    title = f"Distribution of Number of Authors Over 2-Year Ranges\n for {area}" if area else "Distribution of Number of Authors Over 2-Year Ranges"
    plt.title(title)
    plt.xlabel('2-Year Range')
    plt.ylabel('Percentage (%)')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    
    # Show legend
    plt.legend(title='Number of Authors\n(Intervals)', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Step 9: Display the plot
    plt.show()
plot_author_distribution(df_main, area='artificial intelligence')

### ------------------------------------------------------------------------------
# Filter Rows Related to a specfic super topic like "computer science education" or "computer programming"
target_topic = "computer science education"
# region
df = df_main#.sample(n=100,random_state=1)

start_time = time.time()

# Define the specific topic to track

# Empty list to store the matched rows 
education_topic_count = []

# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    try:
        # Check if 'enhanced' is not NaN (missing value) and not empty
        if pd.notna(row['enhanced']) and row['enhanced'].strip():
            # Split the 'enhanced' column into a list of topics
            topics = row['enhanced'].split(', ')
        
        # Check if the target topic is in the list of topics
        if target_topic in [topic.strip() for topic in topics]:
            # Append the result to the list with year, topic, and fractional count
            education_topic_count.append({'year': row['year'], 'enhanced': target_topic, 'fraction': 1})
    except Exception as e:
        print(e)
        break

# Convert the list to a DataFrame
education_df = pd.DataFrame(education_topic_count)

# Calculate the sum of fractions for "Computer Science Education" per year
education_sum_per_year = education_df.groupby(['year', 'enhanced'])['fraction'].sum().reset_index()

# Total elapsed time
total_time = time.time()
print(f"Running on a file with:\n{len(df.index)} number of rows took:\n{total_time - start_time:.2f} seconds")
# endregion

LinePlot(education_sum_per_year,None,None,None,("For super topic: ",target_topic,""))



### Distribution_of_Conf_and_Publ_by_CORE23

# Step 1: Get unique count of 'venue_acronym' and 'doi' based on their 'rank'
unique_venue_count = df_main.groupby('rank')['venue_acronym'].nunique()
unique_doi_count = df_main.groupby('rank')['doi'].nunique()

# Step 2: Calculate the total number of unique 'venue_acronym' and 'doi'
total_unique_venues = df_main['venue_acronym'].nunique()
total_unique_dois = df_main['doi'].nunique()

# Step 3: Calculate the percentages
venue_percentage = (unique_venue_count / total_unique_venues * 100)
doi_percentage = (unique_doi_count / total_unique_dois * 100)

rank_order = ['A*', 'A', 'B', 'C']

# Step 4: Generate the bar chart
ranks = rank_order
bar_width = 0.35  # width of the bars
index = range(len(ranks))  # x locations for each group

# Plot
fig, ax = plt.subplots(figsize=(8, 5))

# Bar for the venue_acronym percentage (White bar)
bar1 = ax.bar(index, venue_percentage.reindex(rank_order), bar_width, label="Conferences w. CORE2023 ranking", color='white', edgecolor='black', align='center')
# Bar for the DOI percentage (Black bar)
bar2 = ax.bar([i + bar_width for i in index], doi_percentage.reindex(rank_order), bar_width, label="Published Papers in Dataset", color='black', align='center')

# Step 5: Label and format the plot
ax.set_xlabel('Rank')
ax.set_ylabel('Percentage (%)')
ax.set_title('Distribution of Conferences w. CORE2023 Ranking and\n their Legacy Publications in the Dataset')
ax.set_xticks([i + bar_width / 2 for i in index])
ax.set_xticklabels(ranks)  # Set x-ticks to the ranks
ax.legend()

# Show the plot
plt.savefig('Distribution_of_Conf_and_Publ_by_CORE23.pdf')
#plt.show()



# Find avg number of superTopics
# Function to count valid supertopics, handling NaN and non-string values
def count_super_topics(enhanced_str):
    if pd.isna(enhanced_str):  # Check for NaN
        return 0  # If NaN, return 0 valid topics
    
    topics = [topic.strip() for topic in enhanced_str.split(',')]
    # Filter only the topics that are in the superTopics list
    valid_topics = [topic for topic in topics if topic.lower() in [t.lower() for t in superTopics]]
    return (valid_topics)

# Apply the function to the "enhanced" column and count supertopics
df_main['num_super_topics'] = df_main['enhanced'].apply(count_super_topics)

# Calculate the average number of supertopics
average_super_topics = df_main['num_super_topics'].mean()

print(f"Average number of supertopics: {average_super_topics}")


# Count the rows where the number of supertopics is exactly 1
rows_with_one_super_topic = df_main[df_main['num_super_topics'] == 1].shape[0]

print(f"Number of rows with exactly one supertopic: {rows_with_one_super_topic}")

total_rows = df_main.shape[0]

# Calculate the percentage of rows with exactly one supertopic
percentage_one_super_topic = (rows_with_one_super_topic / total_rows) * 100

print(f"Percentage of rows with exactly one supertopic: {percentage_one_super_topic:.2f}%")


from collections import Counter

# Filter rows with exactly one supertopic
rows_with_one_super_topic = df_main[df_main['num_super_topics'] == 1]

# Extract the topics from these rows (assuming each row has one topic)
single_topic_list = df_main['num_super_topics'].apply(lambda x: x[0])

# Filter out topics that are not in the superTopics list
valid_single_topics = [topic for topic in single_topic_list if topic.lower() in [t.lower() for t in superTopics]]

# Count the occurrences of each valid topic
topic_counts = Counter(valid_single_topics)

# Convert the counts to a dictionary
topic_counts_dict = dict(topic_counts)

print(topic_counts_dict)



import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# The sorted dictionary
data = {
    'Computer Systems': 9696, 
    'Artificial Intelligence': 9162, 
    'Computer Imaging and Vision': 3153, 
    'Theoretical Computer Science': 3321, 
    'Robotics': 2749, 
    'Computer Programming': 3040, 
    'Software Engineering': 2054, 
    'Human Computer Interaction': 1967, 
    'Information Retrieval': 929, 
    'Internet': 751, 
    'Computer Security': 784, 
    'Computer Hardware': 1299, 
    'Software': 293, 
    'Computer Networks': 349, 
    'Computer Aided Aesign': 371, 
    'Data Mining': 238, 
    'Information Technology': 227, 
    'Bioinformatics': 192, 
    'Operating Systems': 12
}

# Convert the dictionary to a pandas DataFrame
import pandas as pd
df = pd.DataFrame(list(data.items()), columns=['Super Topic', 'Count'])

# Create the bar chart using seaborn
plt.figure(figsize=(10,6))
sns.barplot(data=df, x='Count', y='Super Topic', palette='Blues_d')

# Add grid
plt.grid(True, axis='x', linestyle='--', alpha=0.7)

# Add labels and title
plt.xlabel('Count')
plt.ylabel('Super Areas')
plt.title('Area Distribution of Publications with only One Super Area')

# Adjust the layout to make sure labels fit
plt.tight_layout()

# Show the plot
plt.show()









percentages = [
    14.85, 13.65, 13.62, 13.43, 13.17, 13.07, 12.66, 12.65, 12.51, 
    12.48, 12.30, 12.21, 12.19, 12.05, 11.81, 11.73, 10.94, 10.80
]

# Calculate the average
average_percentage = sum(percentages) / len(percentages)
average_percentage



