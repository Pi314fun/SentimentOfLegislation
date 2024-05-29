### LawPreprocess Documentation (LawPreprocessDoc.md)

# Documentation for LawPreprocess.ipynb

## Overview

This Jupyter Notebook, `LawPreprocess.ipynb`, outlines the preprocessing steps required to clean and prepare legislative bill data for sentiment analysis and embedding generation. The final goal is to use this data in a PyTorch Geometric graph-based model to predict bill sentiment based on tweet sentiment. The notebook follows a structured approach to ensure explainability, reproducibility, and high-quality data processing.

## Table of Contents

1. [Data Loading and Initial Exploration](#data-loading-and-initial-exploration)
2. [Text Preprocessing](#text-preprocessing)
3. [Removing Unnecessary Columns and Formatting Dates](#removing-unnecessary-columns-and-formatting-dates)
4. [Combining and Cleaning Text Data](#combining-and-cleaning-text-data)
5. [Generate Embeddings](#generate-embeddings)
6. [Identify and Remove Duplicates Using Embeddings](#identify-and-remove-duplicates-using-embeddings)
7. [Dimensionality Reduction and Normalization](#dimensionality-reduction-and-normalization)
8. [Final Data Preparation](#final-data-preparation)
9. [Sentiment Analysis](#sentiment-analysis)

## Data Loading and Initial Exploration

**Description**:
This step involves loading the bill dataset, displaying data types, summary statistics, and identifying missing values to understand the data structure and quality.

**Code**:

```python
import pandas as pd
import numpy as np
import torch
from transformers import RobertaModel, RobertaTokenizer
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from multiprocessing import Pool, cpu_count
import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset
bills_df = pd.read_csv('BillData/refined_detailed_bills.csv')

# Display data types
print("Data Types:")
print(bills_df.dtypes)
print("\n")

# Display summary statistics for numerical columns
print("Summary Statistics:")
print(bills_df.describe())
print("\n")

# Identify missing values
print("Missing Values:")
missing_values = bills_df.isnull().sum()
print(missing_values)
print("\n")

# Display the first few rows of the dataframe to understand the data format
print("Data Format (first few rows):")
print(bills_df.head())

print(f"Number of bills: {bills_df.shape[0]}")
```

## Text Preprocessing

**Description**:
This step preprocesses the text data by converting to lowercase, removing non-word characters, and applying tokenization and lemmatization to standardize it for further analysis.

**Code**:

```python
def safe_json_loads(s):
    s = re.sub(r"([{|,]\s*'?)(\w+)'?\s*:", r'\1"\2":', s)  # Fix keys
    s = re.sub(r":\s*'([^']+)'(\s*[},])", r': "\1"\2', s)  # Fix values
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return {}

def clean_text(text):
    return text.str.lower().str.replace(r'\W', ' ', regex=True)

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in tokens])

print("Starting to load data...")
bills_df = pd.read_csv('BillData/refined_detailed_bills.csv')
print("Data loaded successfully.")
```

## Removing Unnecessary Columns and Formatting Dates

**Description**:
This step removes unnecessary columns and converts the `status_date` column to a standardized date format.

**Code**:

```python
# Drop unnecessary columns
bills_df.drop(['state_link', 'sponsors', 'votes', 'url'], axis=1, inplace=True)

print("Converting date columns...")
bills_df['status_date'] = pd.to_datetime(bills_df['status_date'])
print("Conversion completed.")
```

## Combining and Cleaning Text Data

**Description**:
This step combines the `title` and `description` columns into a single text field, cleans the text, and applies preprocessing.

**Code**:

```python
print("Combining title and description...")
bills_df['full_text'] = bills_df['title'] + ' ' + bills_df['description']
bills_df['full_text'] = clean_text(bills_df['full_text'])
print("Text combined and cleaned.")

print("Applying text preprocessing...")
bills_df['processed_text'] = bills_df['full_text'].apply(preprocess_text)
print("Text preprocessing completed.")

# Create and save state mapping
state_map = dict(enumerate(bills_df['state'].unique()))
json.dump(state_map, open('BillData/state_map.json', 'w'))
print("State map saved to 'BillData/state_map.json'.")

print("Saving processed data...")
bills_df.to_csv('BillData/final_processed_bills.csv', index=False)
print("Data saved successfully to 'BillData/final_processed_bills.csv'.")
```

## Generate Embeddings

**Description**:
This step generates text embeddings using a pre-trained RoBERTa model. The embeddings capture the semantic meaning of the bills, which are essential for graph-based ML models.

**Code**:

```python
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
        print("Loading dataset...")
        bills_df = pd.read_csv('BillData/final_processed_bills.csv')

        if 'processed_text' not in bills_df.columns:
            raise ValueError("Processed text column not found in the dataset.")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        print("Loading RoBERTa model and tokenizer...")
        model = RobertaModel.from_pretrained('roberta-base')
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

        texts = bills_df['processed_text'].tolist()

        print("Generating embeddings...")
        embeddings = get_embeddings(model, tokenizer, texts, device=device)
        print("Embeddings generation complete.")

        embeddings_file = "BillData/roberta_bills_embeddings.npy"
        print("Saving embeddings...")
        save_embeddings(embeddings, embeddings_file)
        print("Process completed successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")
```

## Identify and Remove Duplicates Using Embeddings

**Description**:
This step identifies and removes duplicate bills based on cosine similarity of their embeddings, ensuring that only unique bills are retained.

**Code**:

```python
from scipy.spatial.distance import cdist

embeddings = np.load("BillData/RoBERTa_bills_embeddings.npy")
bills_df = pd.read_csv('BillData/final_processed_bills.csv')

assert len(embeddings) == len(bills_df), "Mismatch between embeddings and DataFrame length."
cosine_similarities = 1 - cdist(embeddings, embeddings, metric='cosine')
similarity_threshold = 0.99999999

def identify_duplicates(similarity_matrix, threshold):
    num_bills = similarity_matrix.shape[0]
    duplicates = set()

    for i in range(num_bills):
        for j in range(i + 1, num_bills):
            if similarity_matrix[i, j] > threshold:
                duplicates.add(j)

    return list(duplicates)

duplicate_indices = identify_duplicates(cosine_similarities, similarity_threshold)
bills_df_no_duplicates = bills_df.drop(index=duplicate_indices).reset_index(drop=True)
embeddings_no_duplicates = np.delete(embeddings, duplicate_indices, axis=0)
bills_df_no_duplicates.to_csv('BillData/final_bills_no_duplicates.csv', index=False)
np.save("BillData/RoBERTa_bills_embeddings_no_duplicates.npy", embeddings_no_duplicates)

print(f"Removed {len(duplicate_indices)} duplicate bills. Cleaned data and embeddings saved successfully.")
```

## Dimensionality Reduction and Normalization

**Description**:
This step applies Principal Component Analysis (PCA) to the embeddings for

 dimensionality reduction and normalizes the embeddings.

**Code**:

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

bill_embeddings = np.load("BillData/RoBERTa_bills_embeddings_no_duplicates.npy")
bills_df = pd.read_csv('BillData/final_bills_no_duplicates.csv')

assert len(bill_embeddings) == len(bills_df), "Mismatch between bill embeddings and DataFrame length."

scaler = StandardScaler()
bill_embeddings_scaled = scaler.fit_transform(bill_embeddings)

pca = PCA(n_components=0.95)
bill_embeddings_pca = pca.fit_transform(bill_embeddings_scaled)

np.save("BillData/roberta_bills_embeddings_pca.npy", bill_embeddings_pca)
print(f"Updated bill embeddings saved to 'BillData/roberta_bills_embeddings_pca.npy'.")

bills_df['pca_embeddings'] = list(bill_embeddings_pca)
bills_df.to_csv('BillData/bills.csv', index=False)

print("DataFrame with PCA embeddings saved successfully.")
```

## Final Data Preparation

**Description**:
This step prepares the final dataset by removing unnecessary columns, formatting dates, and adding placeholders for sentiment columns.

**Code**:

```python
bills_df = pd.read_csv('BillData/bills.csv')

columns_to_drop = ['bill_number', 'title', 'description', 'url', 'state_link', 'state', 'current_body_id', 'sponsors', 'subjects', 'texts', 'votes', 'processed_text', 'full_text']
existing_columns_to_drop = [col for col in columns_to_drop if col in bills_df.columns]
bills_df.drop(columns=existing_columns_to_drop, inplace=True)

bills_df['status_date'] = pd.to_datetime(bills_df['status_date']).dt.strftime('%Y-%m-%d')

bills_df['positive'] = 0.0
bills_df['neutral'] = 0.0
bills_df['negative'] = 0.0

output_path = 'BillData/bills.csv'
bills_df.to_csv(output_path, index=False)

print(f"DataFrame saved to {output_path}")
print(bills_df.head())
```

## Sentiment Analysis

**Description**:
This step performs sentiment analysis using a pre-trained sentiment analysis model on the bill texts and appends the sentiment scores to the DataFrame.

**Code**:

```python
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

def load_sentiment_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        print("Sentiment model and tokenizer loaded successfully.")
        return tokenizer, model
    except Exception as e:
        print(f"Error loading sentiment model or tokenizer: {e}")
        raise

def sentiment_analysis(texts, tokenizer, model, batch_size=16, device='cpu'):
    try:
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
    except Exception as e:
        print(f"Error performing sentiment analysis: {e}")
        raise

def load_bills(file_path):
    try:
        bills_df = pd.read_csv(file_path)
        print(f"Loaded {len(bills_df)} bills from {file_path}.")
        return bills_df
    except FileNotFoundError as e:
        print(f"Error loading file: {e}")
        raise
    except pd.errors.ParserError as e:
        print(f"Error parsing file: {e}")
        raise

def apply_sentiment_analysis_to_bills(bills_df, tokenizer, model, batch_size=16, device='cpu'):
    try:
        texts = bills_df['processed_text'].tolist()
        sentiments = sentiment_analysis(texts, tokenizer, model, batch_size, device)
        sentiments_df = pd.DataFrame(sentiments.numpy(), columns=['bill_positive', 'bill_neutral', 'bill_negative'])
        bills_df = pd.concat([bills_df, sentiments_df], axis=1)
        print("Sentiment analysis applied to all bills.")
        return bills_df, sentiments
    except Exception as e:
        print(f"Error applying sentiment analysis: {e}")
        raise

def save_updated_bills_dataframe(bills_df, file_path):
    try:
        bills_df.to_csv(file_path, index=False)
        print(f"Updated DataFrame saved to {file_path}.")
    except Exception as e:
        print(f"Error saving updated DataFrame: {e}")
        raise

def save_tensor(tensor, file_path):
    try:
        torch.save(tensor, file_path)
        print(f"Tensor saved to {file_path}.")
    except Exception as e:
        print(f"Error saving tensor: {e}")
        raise

def verify_saved_file(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Verification successful. First few rows of {file_path}:")
        print(df.head())
    except Exception as e:
        print(f"Error verifying saved file: {e}")
        raise

if __name__ == "__main__":
    try:
        # Load the sentiment analysis model and tokenizer
        tokenizer, model = load_sentiment_model()

        # Load the bills DataFrame
        bills_df = load_bills('BillData/final_bills_no_duplicates.csv')

        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # Apply sentiment analysis to the DataFrame
        bills_df, sentiments_tensor = apply_sentiment_analysis_to_bills(bills_df, tokenizer, model, device=device)

        # Save the updated DataFrame
        save_updated_bills_dataframe(bills_df, 'BillData/bills.csv')

        # Save the tensor containing sentiment probabilities
        save_tensor(sentiments_tensor, 'BillData/bills_sentiments_tensor.pt')

        # Verify the saved file
        verify_saved_file('BillData/bills.csv')

    except Exception as e:
        print(f"An error occurred in the main execution block: {e}")
```
