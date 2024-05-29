# Summary of the Pipeline and Processes

## Overview

The `LawPreprocess.ipynb` notebook outlines a comprehensive preprocessing pipeline for legislative bill data to prepare it for sentiment analysis and embedding generation. This data will be used in a PyTorch Geometric graph-based model to predict bill sentiment based on tweet sentiment.

### Steps in the Pipeline

1. **Data Loading and Initial Exploration**:
   - Load the bill dataset and display data types, summary statistics, and missing values.
   - Examine the data format and determine the number of bills.

2. **Text Preprocessing**:
   - Convert text to lowercase.
   - Remove non-word characters and punctuation.
   - Tokenize and lemmatize the text to standardize it for further analysis.
   - Save the preprocessed text to the dataset.

3. **Removing Unnecessary Columns and Formatting Dates**:
   - Drop unnecessary columns.
   - Convert the `status_date` column to a standardized date format.

4. **Combining and Cleaning Text Data**:
   - Combine the `title` and `description` columns into a single text field.
   - Clean the text data.
   - Apply text preprocessing and save the processed text.

5. **Generate Embeddings**:
   - Use a pre-trained RoBERTa model to generate embeddings for the bills.
   - These embeddings capture the semantic meaning of the bills.
   - Save the generated embeddings to a file for efficient loading and use in PyTorch Geometric.

6. **Identify and Remove Duplicates Using Embeddings**:
   - Compute pairwise cosine similarity of the bill embeddings to identify duplicates.
   - Remove duplicate bills based on a predefined similarity threshold.
   - Save the cleaned dataset and updated embeddings.

7. **Dimensionality Reduction and Normalization**:
   - Apply Principal Component Analysis (PCA) to the embeddings for dimensionality reduction.
   - Normalize the embeddings.
   - Save the updated embeddings and DataFrame.

8. **Final Data Preparation**:
   - Remove unnecessary columns.
   - Format dates and add placeholders for sentiment columns.
   - Save the final prepared dataset.

9. **Sentiment Analysis**:
   - Load a pre-trained sentiment analysis model.
   - Perform sentiment analysis on the preprocessed bill texts, producing probabilities for positive, neutral, and negative sentiments.
   - Append the sentiment scores to the dataset.
   - Save the updated dataset with sentiment scores and the tensor containing sentiment probabilities.

### Conclusion

This preprocessing pipeline ensures that the bill data is cleaned, standardized, and enriched with sentiment scores and embeddings, making it suitable for integration into a graph-based machine learning model using PyTorch Geometric. The pipeline's modular design and thorough documentation facilitate future updates and continuous learning, ensuring high accuracy and reliability in predicting bill sentiment based on tweet sentiment.
