import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import tqdm as tqdm
import re


def load_and_preprocess_data(file_path):
   # Loading DataFrame
    print("loading df")
    df = pd.read_csv(file_path)

    # Cleaning
    print("cleaning")
    df['title'] = df['title'].str.lower()
    df['title'].fillna('', inplace=True)
    df['title'] = df['title'].apply(lambda x: re.sub("[^\\w\\s]", "", x))

    # Tokenize the titles into text
    print("tokenizing")
    df['tokens'] = df['title'].apply(word_tokenize)
    df['text'] = df['tokens'].apply(lambda x: ' '.join(x))

    return df


def perform_clustering(category_df, vectorizer, num_clusters):
    X = vectorizer.fit_transform(category_df['text'])
    kmeans = KMeans(n_clusters=num_clusters, n_init=2)
    category_df['cluster'] = kmeans.fit_predict(X)

# Load CSV into DataFrame
df = load_and_preprocess_data('amazon_products.csv')

# TF-IDF vectorization
vectorizer = TfidfVectorizer()

print("Grouping")
# Organize categories into groups
grouped_by_category = df.groupby('category_id')
category_dict = {category_id: group for category_id, group in grouped_by_category}

# Arbitrary number of clusters
num_clusters = 5

# Apply K-Means clustering for each category
print("vectorize + clustering")
for category_id, category_df in category_dict.items():
    print(category_id)
    perform_clustering(category_df, vectorizer, num_clusters)

# # Display the results
# for category_id, category_df in category_dict.items():
#     print(f"Category ID: {category_id}")
#     print(category_df[['title', 'cluster']])
#     print("\n")
