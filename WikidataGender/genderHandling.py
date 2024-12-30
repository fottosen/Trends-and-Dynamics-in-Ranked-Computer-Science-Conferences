from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Clean the name wikiData query names and save gender
'''
# 1. Read the CSV into a DataFrame
df = pd.read_csv("wikiDataGender/DBLP_authors_with_gender_nameGen.csv")

# 2. Define rules for genderLabel transformation
def transform_gender_label(gender):
    if gender == "cisgender man":
        return "male"
    elif gender in ["non-binary", "bigender", "intersex", "intersex man",
                    "muxe", "trans man", "trans woman", "transfeminine"]:
        return "other"
    return gender  # Leave all other genders unchanged

# Apply the transformation function to the genderLabel column
df['genderLabel'] = df['genderLabel'].apply(transform_gender_label)

# 3. Filter out rows with undesired gender labels
unwanted_genders = ['eunuch', 'gender unknown', 'undisclosed gender']
df = df[~df['genderLabel'].isin(unwanted_genders)]

# 4. Group by 'name' and check if all gender labels are the same for each name
consistent_df = df.groupby('name').filter(lambda x: len(x['genderLabel'].unique()) == 1)

# 5. Drop duplicate entries, since the gender label would be the same for consistent groups
consistent_df = consistent_df.drop_duplicates(subset=['name', 'genderLabel'])

# 6. Create the final DataFrame with "name" and the consistent "genderLabel"
final_df = consistent_df[['name', 'genderLabel']]

# 7. Save the result to a new CSV or use it for further analysis
final_df.to_csv("wikiDataGender/DBLP_authors_with_gender_nameGen_cleaned.csv", index=False)
'''

## Add Gender data from wikidata in the following order: wikiq --> gender, and name --> gender
'''
# Step 1: Load CSV Files
gender_df = pd.read_csv('wikiDataGender/DBLP_authors_with_gender.csv') 
gender_df=gender_df.drop_duplicates()
gender1_2_df = pd.read_csv('wikiDataGender/DBLP_authors_with_gender1_2.csv')
name_gender_cleaned_df = pd.read_csv('wikiDataGender/DBLP_authors_with_gender_nameGen_cleaned.csv')

# Step 2: Remove "https://dblp.org/pid/" from the dblpPID column in DBLP_authors_with_gender.csv
gender_df['dblpPID'] = gender_df['dblpPID'].str.replace('https://dblp.org/pid/', '')

# Step 3: Merge genderLabel from DBLP_authors_with_gender1_2.csv into DBLP_authors_with_gender.csv based on dblpPID
# Perform a left join. Preferring the genderLabel in DBLP_authors_with_gender1_2.csv if available.
gender_df = pd.merge(gender_df, gender1_2_df[['dblpPID', 'genderLabel']], on='dblpPID', how='left', suffixes=('', '_from1_2'))

# Combine genderLabel from both datasets into the first genderLabel column, preferring the one from the merged column if available.
gender_df['genderLabel'] = gender_df['genderLabel'].combine_first(gender_df['genderLabel_from1_2'])

# Drop the helper column
gender_df.drop(columns=['genderLabel_from1_2'], inplace=True)

# Step 4: For rows where genderLabel is still missing, try to fill it using the name match from DBLP_authors_with_gender_nameGen_cleaned.csv
# Left merge the datasets based on the 'name' column
gender_df = pd.merge(gender_df, name_gender_cleaned_df[['name', 'genderLabel']], on='name', how='left', suffixes=('', '_fromNameGen'))

# Again, fill empty genderLabel with the gender from the nameGen file if there was no value.
gender_df['genderLabel'] = gender_df['genderLabel'].combine_first(gender_df['genderLabel_fromNameGen'])

# Drop the helper column
gender_df.drop(columns=['genderLabel_fromNameGen'], inplace=True)

# Step 5: Save the result to a new CSV file
gender_df.to_csv('wikiDataGender/DBLP_authors_w_wikidata.csv', index=False)

# Optional: Print the first few rows to verify the results
print(gender_df.head())
'''

## Add rank to this file
'''
# Step 1: Load CSV Files
# Load 'DBLP_authors_w_wikidata.csv' (which already contains dblpPID, name, genderLabel)
authors_df = pd.read_csv('wikiDataGender/DBLP_authors_w_wikidata.csv')
authors_df = authors_df.drop_duplicates(subset=["dblpPID", "doi"]) # Only one person pr article

# Load 'df_w_topics_#ofAuthors.csv' which contains rank and doi columns
topics_df = pd.read_csv('df_w_topics_#ofAuthors.csv')

# Step 2: Merge DataFrames on 'doi'
# Using a left join to only keep the rows from authors_df and add 'rank' from topics_df
merged_df = pd.merge(authors_df, topics_df[['doi', 'rank']], on='doi', how='left', validate='many_to_one')

# Step 3: Save the final DataFrame to a new CSV
merged_df.to_csv('wikiDataGender/DBLP_authors_w_wikidata_ranks.csv', index=False)

# Optional: Print the first few rows to verify the result
#print(merged_df.head())
'''

## Add topic and numberOfCreators for each doi/author
'''
# Load df_w_topics_#ofAhuthors.csv
df_topics = pd.read_csv('df_w_topics_#ofAuthors.csv')

# Load DBLP_authors_w_wikidata_ranks.csv
df_dblp = pd.read_csv('wikiDataGender/DBLP_authors_w_wikidata_ranks.csv')

# Select only relevant columns from df_topics: doi, filtered_enhanced, and numberOfCreators
df_topics_filtered = df_topics[['doi', 'enhanced', 'numberOfCreators']]

# Merge the DBLP file with the topics data, based on 'doi'
merged_df = pd.merge(df_dblp, df_topics_filtered, how='left', on='doi')

merged_df = merged_df.drop_duplicates(subset=["dblpPID", "doi"]) # Only one person pr article

# Save the extended dataframe to a new CSV file
merged_df.to_csv('wikiDataGender/DBLP_authors_w_wikidata_ranks_topics.csv', index=False)
'''

## Test piechart plot
'''
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the merged CSV dataset
df = pd.read_csv('wikiDataGender/DBLP_authors_w_wikidata_ranks_topics.csv')

# Step 2: Pie Chart of Overall Gender Distribution
overall_gender_counts = df['genderLabel'].value_counts()

# Creating the overall gender distribution pie chart
plt.figure(figsize=(6, 6))  # Size of the pie chart
plt.pie(
    overall_gender_counts, 
    labels=overall_gender_counts.index, 
    autopct='%1.1f%%', 
    startangle=90, 
    colors=['#ff9999','#66b3ff','#99ff99'] # Customize your colors as needed
)
plt.title('Overall Gender Distribution')
plt.show()



# Filter out the smaller groups
small_gender_groups = df[~df['genderLabel'].isin(['male', 'female'])]

# Get the counts for those smaller groups
small_gender_counts = small_gender_groups['genderLabel'].value_counts()

# Plot the pie chart for only the smaller groups
plt.pie(
    small_gender_counts, 
    labels=small_gender_counts.index,
    autopct='%1.1f%%', 
    startangle=90,
    colors=['#ff9999','#66b3ff']
)
plt.title('Distribution for Smaller Gender Groups\nUnder 0.00 percent of total')
plt.show()


# Step 3: Create Pie Charts for Gender Distribution Per Rank

# Specify the ranks of interest
ranks = ['A*', 'A', 'B', 'C']

# Create a pie chart for each rank
for rank in ranks:
    rank_data = df[df['rank'] == rank]  # Filter the data for the current rank
    gender_counts = rank_data['genderLabel'].value_counts()  # Get gender counts for this rank
    
    # Plot Pie Chart for this rank's gender distribution
    plt.figure(figsize=(6, 6))
    plt.pie(
        gender_counts, 
        labels=gender_counts.index, 
        autopct='%1.1f%%', 
        startangle=90,
        colors=['#ff9999','#66b3ff','#99ff99']  # Customize your colors as needed
    )
    plt.title(f'Gender Distribution for Rank: {rank}')
    plt.show()
'''

## Piechart of gender over specfic topic
'''
# Super Topics list
# "artificial intelligence", "bioinformatics", "computer aided design", 
# "computer hardware", "computer imaging and vision", "computer networks", 
# "computer programming", "computer security", "computer systems", 
# "data mining", "human computer interaction", "information retrieval", 
# "information technology", "internet", "operating systems", 
# "robotics", "software", "software engineering", "theoretical computer science"

df = pd.read_csv("wikiDataGender/DBLP_authors_w_wikidata_ranks_topics.csv")

# Function to plot gender distribution for a topic
def plot_gender_distribution_by_topic(topic):
    topic = topic.lower()
    
    #Filter rows where the 'topics' column contains the desired topic
    topic_filter = df["enhanced"].str.contains(topic, regex=False, na=False)
    filtered_df = df[topic_filter]
    
    if filtered_df.empty:
        print(f"No data found for topic: {topic}")
        return
    
    # Group by 'gender' column and count occurrences
    gender_counts = filtered_df["genderLabel"].value_counts()
    
    #Plot a pie chart of the gender distribution
    plt.figure(figsize=(7, 7))
    plt.pie(
        gender_counts, 
        labels=gender_counts.index, 
        autopct='%1.1f%%', 
        startangle=90, 
        colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'],
    )
    
    plt.title(f"Gender Distribution for topic '{topic}'")
    plt.axis('equal')  # Equal aspect ratio ensures that the pie is drawn as a circle
    
    # Show the pie chart
    plt.show()

def plot_gender_distribution_by_topic(topic, exclude_topic=None):
    topic = topic.lower()
    
    # Filter for rows where 'genderLabel' is 'male' or 'female' only
    gender_filter = df['genderLabel'].isin(['male', 'female'])
    topic_filter = df["enhanced"].str.contains(topic, case=False, regex=False, na=False)
    filtered_df = df[gender_filter & topic_filter]
    
    # If an exclude_topic is provided, filter out rows where the 'topics' column contains the exclude topic
    if exclude_topic:
        exclude_topic = exclude_topic.lower()
        exclude_topic_filter = ~filtered_df["enhanced"].str.contains(exclude_topic, case=False, regex=False, na=False)
        filtered_df = filtered_df[exclude_topic_filter]
    
    # Check if the filtered DataFrame is empty after applying filters
    if filtered_df.empty:
        print(f"No data found for topic: {topic}")
        return
    
    # Group by 'gender' column and count occurrences
    gender_counts = filtered_df["genderLabel"].value_counts()
    
    # Plot a pie chart of the gender distribution
    plt.figure(figsize=(7, 7))
    plt.pie(
        gender_counts, 
        labels=gender_counts.index, 
        autopct='%1.1f%%', 
        startangle=90, 
        colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'],
    )
    
    # Add extra title fore exclude topic
    add_topic = "\n all instances and children of topic '{exclude_topic}' are excluded" if exclude_topic else add_topic=""
    plt.title(f"Gender Distribution for topic '{topic}'{add_topic}")
    plt.axis('equal')  # Equal aspect ratio ensures that the pie is drawn as a circle
    
    # Show the pie chart
    plt.show()


plot_gender_distribution_by_topic(topic="computer programming",exclude_topic="computer science education")
plot_gender_distribution_by_topic(topic="computer programming", exclude_topic=None)
'''

## See gender over time

# 1. Read the CSV into a DataFrame
df = pd.read_csv("wikiDataGender/DBLP_authors_w_wikidata_ranks_topics.csv")

# Load the CSV files into pandas dataframes
df_years = pd.read_csv("df_w_topics_#ofAuthors.csv")

# Filter out rows where genderLabel is missing (NaN)
df_authors_filtered = df.dropna(subset=["genderLabel"])

# Merge the two dataframes on the 'doi' column
merged_df = pd.merge(df_authors_filtered, df_years, on="doi")

# Fail safe duplicate removal
merged_df = merged_df.drop_duplicates(subset=["dblpPID", "doi"])

merged_df['year'] = merged_df['year'].astype(int)

### StackedBarPlot of Male/Female procentage over year interval
'''
def StackedBarPlotIntervalGender(df, year1=1967, year2=2023, interval=2, title=None):
    # Filter data for the year range
    df_filtered = df[
        (df['year'] >= year1) &
        (df['year'] <= year2) &
        (df['genderLabel'].isin(['male', 'female']))  # Keep only 'Male' and 'Female'
    ]
    
    # Create a new column for grouping years by a specified interval
    df_filtered['year_interval'] = ((df_filtered['year'] - year1) // interval) * interval + year1
    df_filtered['year_interval'] = df_filtered['year_interval'].astype(int)

    # Create a label for each interval group like '1999-2000'
    df_filtered['year_label'] = df_filtered.apply(
        lambda row: f"{row['year_interval']}-{row['year_interval'] + interval - 1}", axis=1
    )

    # Calculate total counts per interval and gender
    df_grouped = df_filtered.groupby(['year_label', 'genderLabel']).size().reset_index(name='count')

    # Total authors per year interval
    df_grouped['total_per_interval'] = df_grouped.groupby('year_label')['count'].transform('sum')

    # Calculate the percentage share of gender for each interval
    df_grouped['percentage'] = (df_grouped['count'] / df_grouped['total_per_interval']) * 100

    # Pivot the data for plotting
    df_pivot = df_grouped.pivot(index='year_label', columns='genderLabel', values='percentage')

    # Plotting the stacked bar plot
    ax = df_pivot.plot(kind='bar',stacked=True, figsize=(10, 6), cmap='Spectral', width=0.85)

    # Add title and labels
    plt.title(f'Publication Distribution by Gender ({year1}-{year2}) in {interval}-Year Intervals\n{title}')
    plt.xlabel(f'{interval}-Year Intervals')
    plt.ylabel('Percentage (%)')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Show the legend
    plt.legend(title='Gender', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Display the plot
    plt.show()

# Call the function to create the plot
StackedBarPlotIntervalGender(merged_df, year1=1988, year2=2023, interval=2, title="Male and Female Publication Trends")
'''

### Female regression
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

#See values test
'''
df_filtered = merged_df[
    (merged_df['year'] >= 1988) &
    (merged_df['year'] <= 2023) &
    (merged_df['genderLabel'].isin(['male', 'female']))
]

# Step 2: Group data by individual years and calculate female percentages per year
df_grouped = df_filtered.groupby(['year', 'genderLabel']).size().reset_index(name='count')
df_grouped['total_per_year'] = df_grouped.groupby('year')['count'].transform('sum')
df_grouped['percentage'] = (df_grouped['count'] / df_grouped['total_per_year']) * 100

# Step 3: Pivot data so that we have separate columns for each gender, fill missing values with 0
df_pivot = df_grouped.pivot(index='year', columns='genderLabel', values='percentage').fillna(0)

# Extract the individual years and female percentages
years = df_pivot.index.values
female_percentage = df_pivot['female'].values
'''

def LinearRegFemale(df, year1=1988, year2=2023, title=None, f_percent=None):
    # Step 1: Filter DataFrame for specified years, considering only 'male' and 'female' genders
    df_filtered = df[
        (df['year'] >= year1) &
        (df['year'] <= year2) &
        (df['genderLabel'].isin(['male', 'female']))
    ]

    # Step 2: Group data by individual years and calculate female percentages per year
    df_grouped = df_filtered.groupby(['year', 'genderLabel']).size().reset_index(name='count')
    df_grouped['total_per_year'] = df_grouped.groupby('year')['count'].transform('sum')
    df_grouped['percentage'] = (df_grouped['count'] / df_grouped['total_per_year']) * 100

    # Step 3: Pivot data so that we have separate columns for each gender, fill missing values with 0
    df_pivot = df_grouped.pivot(index='year', columns='genderLabel', values='percentage').fillna(0)

    # Extract the individual years and female percentages
    years = df_pivot.index.values
    female_percentage = df_pivot['female'].values

    # Step 4: Perform Linear Regression
    X = sm.add_constant(years)  # Add constant (intercept) for the regression
    model = sm.OLS(female_percentage, X)  # Ordinary Least Squares regression
    results = model.fit()

    # Extract standard deviation (standard error) and p-value from regression results
    std_err = results.bse[1]  # Standard error of the slope coefficient
    p_value = results.pvalues[1]  # P-value of the slope coefficient

    # Print regression summary (optional)
    print(results.summary())

    # Step 6: Calculate and Print Predicted Year for Given Female Percentage
    if f_percent is not None:
        a = results.params[0]  # Intercept
        b = results.params[1]  # Slope
        if b != 0:
            predicted_year = (f_percent - a) / b
            predicted_year_rounded = round(predicted_year)
            # Check if the predicted year is within the data range
            if year1 <= predicted_year_rounded <= year2:
                print(f"Warning: The predicted year {predicted_year_rounded} falls within the regression range ({year1}-{year2}).")
            print(f"The year when female participation is expected to reach {f_percent}% is estimated to be {predicted_year:.2f}.")
        else:
            print("Slope is zero; cannot compute year based on constant female percentage.")

    # Step 5: Visualize the Female Percentage with Time along with Linear Fit
    plt.figure(figsize=(10, 6))
    plt.scatter(years, female_percentage, label='Female Percentage', color='b')
    
    # Plot the regression line
    plt.plot(years, results.predict(X), label=f'Linear Fit (slope = {results.params[1]:.2f})', color='r', linestyle='--')

    plt.xlabel('Year')
    plt.ylabel('Female Percentage')
    plt.ylim(bottom=0)
    plt.title(title or f'Percentage of Female Authors ({year1}-{year2})')
    plt.legend()
    plt.grid(True)
    plt.show()
    #plt.savefig('all_ranks_relative_linear_reg.pdf')

    return std_err, p_value

# Example usage:
# Define the target female percentage
f_percent = 50  # Replace with your desired percentage

# Call the function with your DataFrame and f_percent
std_err, p_value = LinearRegFemale(merged_df, f_percent=f_percent)

print(f"Standard Error (slope): {std_err}")
print(f"P-value (slope): {p_value}")



'''
def LinearRegFemale(df, year1=1967, year2=2023, interval=2, title=None):
    # Step 1: Filter DataFrame for specified years, considering only 'male' and 'female' genders
    df_filtered = df[
        (df['year'] >= year1) &
        (df['year'] <= year2) &
        (df['genderLabel'].isin(['male', 'female']))
    ]

    # Step 2: Group time into intervals
    df_filtered['year_interval'] = ((df_filtered['year'] - year1) // interval) * interval + year1
    df_filtered['year_interval'] = df_filtered['year_interval'].astype(int)

    # Create a readable label for each interval (like '1999-2000')
    df_filtered['year_label'] = df_filtered.apply(
        lambda row: f"{row['year_interval']}-{row['year_interval'] + interval - 1}", axis=1
    )

    # Step 3: Group data and calculate female percentages by interval
    df_grouped = df_filtered.groupby(['year_label', 'genderLabel']).size().reset_index(name='count')
    df_grouped['total_per_interval'] = df_grouped.groupby('year_label')['count'].transform('sum')
    df_grouped['percentage'] = (df_grouped['count'] / df_grouped['total_per_interval']) * 100

    df_pivot = df_grouped.pivot(index='year_label', columns='genderLabel', values='percentage').fillna(0)

    # Step 4: Fit a linear regression model for female percentage
    year_intervals = np.arange(len(df_pivot.index)).reshape(-1, 1)  # X-axis (index of intervals)
    female_percentages = df_pivot['female'].values  # Y-axis (female percentages)

    model = LinearRegression()
    model.fit(year_intervals, female_percentages)
    female_trend_line = model.predict(year_intervals)

    # Get the slope (a) and intercept (b) from the regression model
    slope = model.coef_[0]  # Slope tells us the rate of increase per interval
    intercept = model.intercept_  # Starting value (intercept) at x=0

    # Step 5: Convert the slope to a percentage increase per interval
    # Since we're working with year intervals, multiply slope by 100 to get percentage
    percentage_increase_per_interval = slope * 100

    # Step 6: Plot the actual female percentages and linear regression line
    plt.figure(figsize=(10, 6))

    # Plot actual female percentages
    plt.plot(df_pivot.index, female_percentages, 'bo-', label='Female Author Percentage', markersize=7)

    # Plot linear regression line (the trend line)
    plt.plot(df_pivot.index, female_trend_line, 'r-', label='Linear Regression (Female Trend)', linewidth=2)

    # Step 7: Annotate the graph with the slope-based message
    annotation_text = f"Female authors growth is {percentage_increase_per_interval:.2f}% per {interval}-year interval"
    plt.text(len(df_pivot.index) // 2, max(female_trend_line), annotation_text, fontsize=12, color='green', ha='center')

    # Step 8: Add labels, title, and legend
    plt.title(f'Female Authors ({year1}-{year2})\n{title}')
    plt.xlabel(f'{interval}-Year Intervals')
    plt.ylabel('Female Author Percentage (%)')
    plt.xticks(rotation=45)  # Rotate x-axis labels for readability

    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Show the plot with the annotation
    plt.tight_layout()
    plt.show()

# Example of how to call the function
LinearRegFemale(merged_df, year1=1988, year2=2023, interval=2, title="Female Author Percentage per Year Interval")
'''

### Total gender amount
'''
def PlotMaleFemaleCounts(df, year1=1967, year2=2023, interval=2, title=None):
    # Step 1: Filter the data for 'male' and 'female' genders in the specified year range
    df_filtered = df[
        (df['year'] >= year1) &
        (df['year'] <= year2) &
        (df['genderLabel'].isin(['male', 'female']))
    ]

    # Step 2: Group by year intervals (e.g., 1980-1981, 1982-1983, etc.)
    df_filtered['year_interval'] = ((df_filtered['year'] - year1) // interval) * interval + year1
    df_filtered['year_interval'] = df_filtered['year_interval'].astype(int)

    df_filtered['year_label'] = df_filtered.apply(
        lambda row: f"{row['year_interval']}-{min(row['year_interval'] + interval - 1, year2)}", axis=1
    )

    # Step 3: Group by 'year_label' and 'genderLabel' to count number of male/female authors per interval
    df_grouped = df_filtered.groupby(['year_label', 'genderLabel']).size().reset_index(name='count')

    # Step 4: Pivot the data: Rows as intervals, Columns as 'male' and 'female' counts
    df_pivot = df_grouped.pivot(index='year_label', columns='genderLabel', values='count').fillna(0)

    # Step 5: Plot Male and Female counts over time
    plt.figure(figsize=(10, 6))

    # Plot counts of male and female authors over the intervals
    plt.plot(df_pivot.index, df_pivot['male'], 'b-', label='Male Authors', marker='o', linewidth=2)
    plt.plot(df_pivot.index, df_pivot['female'], 'r-', label='Female Authors', marker='o', linewidth=2)

    # Add labels, title, and legend
    plt.title(f'Count of Male and Female Authors Over Time ({year1}-{year2})\n{title}')
    plt.xlabel(f'{interval}-Year Intervals')
    plt.ylabel('Number of Authors')
    plt.xticks(rotation=45)

    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Tight layout to prevent clipping
    plt.tight_layout()

    # Display the plot
    plt.show()

# Example of how to call the function
PlotMaleFemaleCounts(merged_df, year1=1978, year2=2023, interval=2, title="Number of Male and Female Authors Over the Years")
'''