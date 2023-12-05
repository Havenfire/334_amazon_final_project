import pandas as pd

# Replace 'your_file.csv' with the actual file path
file_path = 'amazon_products.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Use value_counts() to count the occurrences of each category_id
category_counts = df['category_id'].value_counts()

# Get the category with the most entries
most_common_category = category_counts.idxmax()

# Get the count of entries for the most common category
count_most_common_category = category_counts.max()

print(f"The category with the most entries is {most_common_category} with {count_most_common_category} entries.")
