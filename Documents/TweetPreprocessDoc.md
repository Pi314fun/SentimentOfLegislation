# Documentation for `TweetPreprocess.ipynb`

## Overview

This Jupyter Notebook, `TweetPreprocess.ipynb`, outlines the preprocessing steps required to clean and prepare tweet data for sentiment analysis and embedding generation. The final goal is to use this data in a PyTorch Geometric graph-based model to predict bill sentiment based on tweet sentiment. The notebook follows a structured approach to ensure explainability, reproducibility, and high-quality data processing.

## Table of Contents

1. [Data Loading and Initial Exploration](#data-loading-and-initial-exploration)
2. [Text Preprocessing](#text-preprocessing)
3. [Removing Non-English Tweets, Duplicates, and Advertisements](#removing-non-english-tweets-duplicates-and-advertisements)
4. [Extracting Metrics and Log Normalization](#extracting-metrics-and-log-normalization)
5. [Generate Embeddings](#generate-embeddings)
6. [Identify and Remove Duplicates Using Embeddings](#identify-and-remove-duplicates-using-embeddings)
7. [Sentiment Analysis](#sentiment-analysis)
8. [Dimensionality Reduction and Normalization](#dimensionality-reduction-and-normalization)

## Data Loading and Initial Exploration

**Description**:
This step involves loading the tweet dataset, displaying data types, summary statistics, and identifying missing values to understand the data structure and quality.

**Code**:

```python
# Load the dataset
tweets_df = pd.read_csv('TweetData/combined_tweets_data.csv')

# Display data types
print("Data Types:")
print(tweets_df.dtypes)
print("\n")

# Display summary statistics for numerical columns
print("Summary Statistics:")
print(tweets_df.describe())
print("\n")

# Identify missing values
print("Missing Values:")
missing_values = tweets_df.isnull().sum()
print(missing_values)
print("\n")

# Display the first few rows of the dataframe to understand the data format
print("Data Format (first few rows):")
print(tweets_df.head())

print(f"Number of tweets: {tweets_df.shape[0]}")
```

## Text Preprocessing

**Description**:
This step preprocesses the text data by converting to lowercase, removing URLs, mentions, hashtags, and punctuation. It then tokenizes and lemmatizes the text to standardize it for further analysis.

**Code**:

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove URLs, mentions, and hashtags
    text = re.sub(r'http\S+|www\S+|https\S+|@\w+|#\w+', '', text)
    # Remove non-word characters and tokenize
    tokens = word_tokenize(re.sub(r'\W+', ' ', text))
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Apply preprocessing to the tweet column
tweets_df['processed_tweet'] = tweets_df['text'].apply(preprocess_text)
tweets_df.to_csv('TweetData/combined_tweets_data.csv', index=False)
print(tweets_df.head)
```

## Removing Non-English Tweets, Duplicates, and Advertisements

**Description**:
This step removes duplicate tweets, non-English tweets, and promotional/advertising tweets based on a predefined set of keywords.

**Code**:

```python
# Remove Duplicate Tweets
tweets_df = tweets_df.drop_duplicates(subset='tweet_id', keep='first')

# Remove Non-English Tweets
tweets_df = tweets_df[tweets_df['lang'] == 'en']

# Remove Advertising/Promotional Tweets
def is_promotional(text):
    promotional_keywords = ['buy now', 'free', 'discount', 'offer', 'sale', 'shop', 'promotion', 'sponsored', 'advertisement', 'ad', 'click here', 'visit our site', 'subscribe', 'check out', 'limited time']
    for keyword in promotional_keywords:
        if keyword in text.lower():
            return True
    return False

# Apply the is_promotional function to filter out promotional tweets
tweets_df = tweets_df[~tweets_df['text'].apply(is_promotional)]
tweets_df.to_csv('TweetData/combined_tweets_data.csv', index=False)

# Display a sample of the cleaned data
print(f"Number of tweets after cleaning: {tweets_df.shape[0]}")
print("Sample of cleaned data:")
print(tweets_df.head())
```

## Extracting Metrics and Log Normalization

**Description**:
This step extracts engagement metrics from the JSON strings in the `public_metrics` column and applies log normalization to these metrics.

**Code**:

```python
import json

def extract_metrics(json_str):
    keys = ['retweet_count', 'reply_count', 'like_count', 'quote_count', 'bookmark_count', 'impression_count']
    try:
        metrics = json.loads(json_str.replace("'", '"'))
        return {key: metrics.get(key, 0) for key in keys}
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}, input: {json_str}")
        return {key: 0 for key in keys}

tweets_df['metrics'] = tweets_df['public_metrics'].apply(extract_metrics)
tweets_df['total_engagement'] = tweets_df['metrics'].apply(lambda x: sum(x.values()))
tweets_df['log_engage'] = np.log1p(tweets_df['total_engagement'])
tweets_df.to_csv('TweetData/combined_tweets_data.csv', index=False)
```

## Generate Embeddings

**Description**:
This step generates text embeddings using a pre-trained RoBERTa model. The embeddings capture the semantic meaning of the tweets, which are essential for graph-based ML models.

**Code**:

```python
import pandas as pd
import numpy as np
import torch
from transformers import RobertaModel, RobertaTokenizer
from torch.utils.data import DataLoader, Dataset
import os

class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

def get_embeddings(model, tokenizer, texts, batch_size=16, device='cpu'):
    dataset = TextDataset(texts)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = model.to(device)
    all_embeddings = []

    for batch_texts in data_loader:
        try:
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()
            all_embeddings.append(embeddings)
        except Exception as e:
            print(f"Error processing batch: {e}")

    return np.vstack(all_embeddings)

def save_embeddings(embeddings, file_name):
    np.save(file_name, embeddings)
    print(f"Embeddings saved successfully to {file_name}.")

if __name__ == "__main__":
    try:
        tweets_df = pd.read_csv('TweetData/combined_tweets_data.csv')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = RobertaModel.from_pretrained('roberta-base')
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        texts = tweets_df['processed_tweet'].tolist()
        embeddings = get_embeddings(model, tokenizer, texts, device=device)
        embeddings_file = "TweetData/roberta_tweets_embeddings.npy"
        save_embeddings(embeddings, embeddings_file)
    except Exception as e:
        print(f"An error occurred: {e}")
```

## Identify and Remove Duplicates Using Embeddings

**Description**:
This step identifies and removes duplicate tweets based on cosine similarity of their embeddings, ensuring that only unique tweets are retained.

**Code**:

```python
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd

embeddings = np.load("TweetData/roberta_tweets_embeddings.npy")
tweets_df = pd.read_csv('TweetData/combined_tweets_data.csv')

assert len(embeddings) == len(tweets_df), "Mismatch between embeddings and DataFrame length."
cosine_similarities = 1 - cdist(embeddings, embeddings, metric='cosine')
similarity_threshold = 0.99999999

def identify_duplicates(similarity_matrix, threshold):
    num_tweets = similarity_matrix.shape[0]
    duplicates = set()

    for i in range(num_tweets):
        for j in range(i + 1, num_tweets):
            if similarity_matrix[i, j] > threshold:
                duplicates.add(j)

    return list(duplicates)

duplicate_indices = identify_duplicates(cosine_similarities, similarity_threshold)
tweets_df_no_duplicates = tweets_df.drop(index=duplicate_indices).reset_index(drop=True)
embeddings_no_duplicates = np.delete(embeddings, duplicate_indices, axis=0)
tweets_df_no_duplicates.to_csv('TweetData/combined_tweets_no_duplicates.csv', index=False)
np.save

("TweetData/roberta_tweets_embeddings_no_duplicates.npy", embeddings_no_duplicates)

print(f"Removed {len(duplicate_indices)} duplicate tweets. Cleaned data and embeddings saved successfully.")
```

## Sentiment Analysis

**Description**:
This step performs sentiment analysis using a pre-trained sentiment analysis model on the tweets and appends the sentiment scores to the DataFrame.

**Code**:

```python
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

def load_sentiment_model():
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    return tokenizer, model

def sentiment_analysis(texts, tokenizer, model, batch_size=16, device='cpu'):
    model = model.to(device)
    all_scores = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        encoded_input = tokenizer(batch_texts, return_tensors='pt', truncation=True, max_length=512, padding=True)
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        with torch.no_grad():
            output = model(**encoded_input)
        scores = torch.nn.functional.softmax(output.logits, dim=-1).cpu()
        all_scores.append(scores)
    return torch.cat(all_scores, dim=0)

def apply_sentiment_analysis(tweets_df, tokenizer, model, batch_size=16, device='cpu'):
    texts = tweets_df['processed_tweet'].tolist()
    sentiments = sentiment_analysis(texts, tokenizer, model, batch_size, device)
    sentiments_df = pd.DataFrame(sentiments.numpy(), columns=['positive', 'neutral', 'negative'])
    tweets_df = pd.concat([tweets_df, sentiments_df], axis=1)
    return tweets_df, sentiments

def save_updated_dataframe(tweets_df, file_path):
    tweets_df.to_csv(file_path, index=False)

def save_tensor(tensor, file_path):
    torch.save(tensor, file_path)

def verify_saved_file(file_path):
    df = pd.read_csv(file_path)
    print(df.head())

if __name__ == "__main__":
    try:
        tokenizer, model = load_sentiment_model()
        tweets_df = load_tweets('TweetData/combined_tweets_no_duplicates.csv')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tweets_df, sentiments_tensor = apply_sentiment_analysis(tweets_df, tokenizer, model, device=device)
        save_updated_dataframe(tweets_df, 'TweetData/roberta_tweets_sentiments.csv')
        save_tensor(sentiments_tensor, 'TweetData/roberta_tweets_sentiments_tensor.pt')
        verify_saved_file('TweetData/roberta_tweets_sentiments.csv')
    except Exception as e:
        print(f"An error occurred in the main execution block: {e}")
```

## Dimensionality Reduction and Normalization

**Description**:
This step applies dimensionality reduction using Principal Component Analysis (PCA) and normalization to the tweet embeddings to ensure they are more distinguished and meaningful for the subsequent graph-based model.

**Code**:

```python
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the tweet embeddings and the original dataset
tweet_embeddings = np.load("TweetData/roberta_tweets_embeddings_no_duplicates.npy")
tweets_df = pd.read_csv('TweetData/combined_tweets_no_duplicates.csv')

# Ensure the embeddings and the DataFrame have the same length
assert len(tweet_embeddings) == len(tweets_df), "Mismatch between tweet embeddings and DataFrame length."

# Standardize the embeddings
scaler = StandardScaler()
tweet_embeddings_scaled = scaler.fit_transform(tweet_embeddings)

# Apply PCA to reduce dimensionality while retaining 95% variance
pca = PCA(n_components=0.95)
tweet_embeddings_pca = pca.fit_transform(tweet_embeddings_scaled)

# Save the updated embeddings
np.save("TweetData/roberta_tweets_embeddings_pca.npy", tweet_embeddings_pca)
print(f"Updated tweet embeddings saved to 'TweetData/roberta_tweets_embeddings_pca.npy'.")

# Update the DataFrame with PCA embeddings
tweets_df['pca_embeddings'] = list(tweet_embeddings_pca)
tweets_df.to_csv('TweetData/updated_tweets_with_pca.csv', index=False)

print("DataFrame with PCA embeddings saved successfully.")
```

## Conclusion

This notebook provides a comprehensive preprocessing pipeline for tweet data, including data loading, cleaning, preprocessing, embedding generation, and sentiment analysis. The outputs from these steps are suitable for integration into a PyTorch Geometric graph-based model for sentiment prediction on legislative bills.