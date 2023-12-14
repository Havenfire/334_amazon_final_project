import pandas as pd

# Assuming your CSV file is named 'your_file.csv'
file_path = 'title_topic_modeling_results.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Keep only the 'title' and 'category_cluster' columns
df_filtered = df[['topic', 'title']]

# Sort the DataFrame by 'category_cluster'
df_sorted = df_filtered.sort_values(by='topic')

# Save the sorted DataFrame to a new CSV file
sorted_file_path = 'title_topic_modeling_results.csv'
df_sorted.to_csv(sorted_file_path, index=False)

# Display the sorted DataFrame
print(df_sorted)
