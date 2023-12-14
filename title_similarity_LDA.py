import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import LdaModel
from gensim.corpora import Dictionary
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

    # Porter stemming
    print("stemming")
    porter = PorterStemmer()
    df['tokens'] = df['tokens'].apply(lambda x: [porter.stem(word) for word in x])

    df['text'] = df['tokens'].apply(lambda x: ' '.join(x))

    return df

def perform_topic_modeling(category_df, dictionary, lda_model):
    corpus = [dictionary.doc2bow(text.split()) for text in category_df['text']]
    category_df['topic'] = [max(lda_model[doc], key=lambda x: x[1])[0] for doc in corpus]

# Load CSV into DataFrame
df = load_and_preprocess_data('amazon_products.csv')

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])

# Organize categories into groups
grouped_by_category = df.groupby('category_id')
category_dict = {category_id: group for category_id, group in grouped_by_category}

# Arbitrary number of topics
num_topics = 10

# Apply LDA topic modeling for each category
print("LDA topic modeling")
for category_id, category_df in category_dict.items():
    print(category_id)
    dictionary = Dictionary(category_df['tokens'])
    corpus = [dictionary.doc2bow(text) for text in category_df['tokens']]
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)
    perform_topic_modeling(category_df, dictionary, lda_model)

# Combine the dictionary back into a single DataFrame
combined_df = pd.concat(category_dict.values(), ignore_index=True)

# Sort the DataFrame by 'topic'
combined_df.sort_values(by='topic', inplace=True)

# Save the combined DataFrame to a CSV file
combined_df.to_csv('title_topic_modeling_results.csv', index=False)

print("Combined DataFrame saved to 'title_topic_modeling_results.csv'")
