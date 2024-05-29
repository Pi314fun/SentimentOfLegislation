# Summary of the Pipeline and Processes

## Overview

The `TweetPreprocess.ipynb` notebook outlines a comprehensive preprocessing pipeline for tweet data to prepare it for sentiment analysis and embedding generation. This data will be used in a PyTorch Geometric graph-based model to predict bill sentiment based on tweet sentiment.

### Steps in the Pipeline

1. **Data Loading and Initial Exploration**:
   - Load the tweet dataset and display data types, summary statistics, and missing values.
   - Examine the data format and determine the number of tweets.

2. **Text Preprocessing**:
   - Convert text to lowercase.
   - Remove URLs, mentions, hashtags, and punctuation.
   - Tokenize and lemmatize the text to standardize it for further analysis.
   - Save the preprocessed text to the dataset.

3. **Removing Non-English Tweets, Duplicates, and Advertisements**:
   - Remove duplicate tweets based on tweet ID.
   - Filter out non-English tweets.
   - Identify and remove promotional or advertising tweets based on specific keywords.
   - Save the cleaned dataset.

4. **Extracting Metrics and Log Normalization**:
   - Extract engagement metrics (e.g., retweet count, like count) from JSON strings.
   - Calculate the total engagement and apply log normalization to these metrics.
   - Save the updated dataset with normalized engagement metrics.

5. **Generate Embeddings**:
   - Use a pre-trained RoBERTa model to generate embeddings for the tweets.
   - These embeddings capture the semantic meaning of the tweets.
   - Save the generated embeddings to a file for efficient loading and use in PyTorch Geometric.

6. **Identify and Remove Duplicates Using Embeddings**:
   - Compute pairwise cosine similarity of the tweet embeddings to identify duplicates.
   - Remove duplicate tweets based on a predefined similarity threshold.
   - Save the cleaned dataset and updated embeddings.

7. **Sentiment Analysis**:
   - Load a pre-trained sentiment analysis model.
   - Perform sentiment analysis on the preprocessed tweets, producing probabilities for positive, neutral, and negative sentiments.
   - Append the sentiment scores to the dataset.
   - Save the updated dataset with sentiment scores and the tensor containing sentiment probabilities.

8. **Dimensionality Reduction and Normalization**:
   - Apply dimensionality reduction using Principal Component Analysis (PCA) to the tweet embeddings.
   - Normalize the embeddings to ensure they are more distinguished and meaningful.
   - Save the updated embeddings and the dataset with PCA embeddings.

### Conclusion

This preprocessing pipeline ensures that the tweet data is cleaned, standardized, and enriched with sentiment scores and embeddings, making it suitable for integration into a graph-based machine learning model using PyTorch Geometric. The pipeline's modular design and thorough documentation facilitate future updates and continuous learning, ensuring high accuracy and reliability in predicting bill sentiment based on tweet sentiment.