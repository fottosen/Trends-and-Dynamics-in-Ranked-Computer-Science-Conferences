import pandas as pd

# Load CSV 
df = pd.read_csv('')


filtered_df = df[df['venue_title'].isin(['International World Wide Web Conference', 'IEEE International Conference on Robotics and Automation', 'ACM International Symposium on Computer Architecture'])]

# Sample 1000 random rows
sampled_df = filtered_df.sample(n=1000, random_state=17)

# Save the sampled data to a new CSV file
sampled_df.to_csv('sampled_file.csv', index=False)