# Graph-Based Sentiment Analysis Pipeline

Welcome to the Graph-Based Sentiment Analysis Pipeline repository. This project provides a comprehensive framework for preprocessing and analyzing tweets and legislative data, applying PCA for dimensionality reduction, and building a graph-based sentiment analysis model using state-of-the-art machine learning techniques.

## Table of Contents

- [Introduction](#introduction)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

This repository contains several Jupyter notebooks and datasets used for preprocessing tweet and legislative data, constructing graphs, and performing sentiment analysis. The main components include data preprocessing, PCA for dimensionality reduction, and sentiment analysis using RoBERTa embeddings.

## Repository Structure

```plaintext
.
├── PCAGraph.ipynb
├── ConnectionChoices.ipynb
├── LawPreprocess.ipynb
├── PipeLine.ipynb
├── TweetPreprocess.ipynb
├── TweetData/
│   ├── combined_tweets_data.csv
│   ├── combined_tweets_no_duplicates.csv
│   ├── final_tweets.csv
│   ├── roberta_tweets_embeddings.npy
│   ├── roberta_tweets_embeddings_no_duplicates.npy
│   ├── roberta_tweets_embeddings_pca.npy
│   ├── roberta_tweets_sentiments.csv
│   ├── roberta_tweets_sentiments_tensor.pt
│   ├── tweets.csv
│   ├── updated_tweets_with_pca.csv
├── PCAGraph_files/
│   ├── PCAGraph_23_1.png
│   ├── PCAGraph_23_10.png
│   ├── PCAGraph_23_2.png
│   ├── PCAGraph_23_3.png
│   ├── PCAGraph_23_4.png
│   ├── PCAGraph_23_5.png
│   ├── PCAGraph_23_6.png
│   ├── PCAGraph_23_7.png
│   ├── PCAGraph_23_8.png
│   ├── PCAGraph_23_9.png
├── Models/
│   ├── all_results.xlsx
│   ├── top_20_results.xlsx
├── GraphData/
│   ├── graph_data.pt
├── Documents/
│   ├── combined_data_collection_pipelines.gv.pdf
│   ├── LawPreprocessDoc.md
│   ├── LawPreprocessSum.md
│   ├── law_preprocessing_pipeline.gv.pdf
│   ├── law_preprocessing_pipeline.png
│   ├── PCAGraphDoc.md
│   ├── PCAGraphSum.md
│   ├── TweetPreprocessDoc.md
│   ├── TweetPreprocessSum.md
│   ├── tweet_preprocessing_pipeline.gv.pdf
├── BillData/
│   ├── aggregated_bill_sentiments.csv
│   ├── bills.csv
│   ├── bills_sentiments_tensor.pt
│   ├── final_bills.csv
│   ├── final_bills_no_duplicates.csv
│   ├── final_processed_bills.csv
│   ├── refined_detailed_bills.csv
│   ├── roberta_bills_embeddings.npy
│   ├── RoBERTa_bills_embeddings_no_duplicates.npy
│   ├── roberta_bills_embeddings_pca.npy
│   ├── state_map.json
│   ├── updated_bills_with_pca.csv
├── X_APIV2.ipynb
├── LegiScan_APIV3.ipynb
```

## Main Files

### LegiScan_APIV3.ipynb

#### LegiScan_APIV3 Summary of the Pipeline and Processes

1. **Loading Configuration and API Key**: The first script loads the `LegiScan_api_key` from a configuration file, preparing the key for subsequent API requests to fetch legislative data.

#### Functions for Law Search and Data Manipulation

2. **API Access and Search Functionality**:
   - `search_laws`: This function queries an API for laws based on a specified keyword and state. The API's response is returned as JSON.
   - `process_search_results`: This function processes the search results by fetching detailed bill information using another function, `fetch_bill_details`, and accumulates this data for further processing.
   - `save_to_file`: It saves JSON data to a specified directory, ensuring that the directory exists.

#### Detailed Bill Fetching and Data Processing

3. **Detailed Bill Information Retrieval**:
   - `fetch_bill_details`: Fetches detailed information for a specific bill ID using the API.
   - `extract_bill_ids`: Extracts bill IDs from JSON files to allow detailed fetching of bill information.
   - `process_files_and_fetch_details`: Processes files containing search results to fetch detailed information about each bill, then saves this consolidated information to a CSV file.

#### Data Cleaning and Refining

4. **Data Cleaning and Exporting**:
   - A series of scripts load, clean, and refine bill data, performing operations like:
     - Converting date strings to datetime objects.
     - Sorting and deduplicating entries based on specific criteria like `status_date` and `session_id`.
     - Filtering DataFrame columns to retain only specified columns, handling missing or additional columns appropriately.
   - The cleaned and refined data is then saved as a new CSV file.

#### Analysis and Debugging

5. **Data Analysis and Debugging**:

   - The final script analyzes the cleaned dataset by:
     - Providing basic dataset information, such as data types, non-null counts, and unique values.
     - Outputting statistical summaries for numeric columns and previewing the dataset structure to understand the data's format.

#### Potential Issues and Recommendations

- **Error Handling**: While your scripts include some basic error handling, consider expanding this to more robustly handle potential API failures or unexpected data formats, especially when working with external data sources.
- **Performance Optimization**: If the dataset or number of API calls grows significantly, consider optimizing your data processing steps or using asynchronous requests for fetching data.
- **Data Quality Checks**: Implement additional checks for data quality, especially post-transformation steps to ensure that the data transformations haven't introduced errors or unexpected mutations in the data.

### X_APIV2.ipynb

#### X_APIV2 Summary of the Pipeline and Processes

#### Twitter API Authentication

1. **Import Libraries**: Importing necessary libraries for handling Twitter API and file operations.
2. **Configuration File Handling**: It sets up a path for a configuration file (`config.json`) which contains API credentials. This path is dynamically constructed using the current directory's parent.
3. **Load Configuration**: The configuration file is opened and its JSON content (API credentials) is loaded into a variable.
4. **Twitter API Authentication**: Using the credentials from the configuration, it authenticates with Twitter using Tweepy’s `OAuthHandler`.
5. **Authentication Verification**: It attempts to verify the credentials. If successful, it prints "Authentication OK". If not, it prints the error message.

#### JSON Structure Printer

1. **Print JSON-Like Structure**: Recursively prints the structure of a JSON-like dictionary or list, showing the nesting and organization of data.
2. **Load and Print Data**: The function is used to load data from a specified JSON file (`tweets_data.json`) and prints its hierarchical structure.

#### JSON Data Consolidation and CSV Export

1. **Import Libraries**: Imports the necessary libraries for handling JSON and dataframes.
2. **Functions Defined**:
    - `read_and_combine_json_files()`: Reads multiple JSON files, extracts data, handles unique identification of tweets, and combines them into a list.
    - `export_to_csv()`: Takes the combined data, converts it into a DataFrame, and exports it to a CSV file.
3. **Process and Export Data**: The functions are called with specified JSON file paths to process and export the data.

#### Data Analysis with Pandas

1. **Load Dataset**: Data is loaded from `combined_tweets_data.csv`.
2. **Data Inspection and Analysis**:
    - Basic information about dataset structure and non-null count.
    - Data types of columns and unique values count.
    - Statistical summary of numeric columns.
3. **Output Descriptions**: It prints a variety of information about the dataset, including missing values, unique counts, and basic statistics.

### TweetPreprocess.ipynb

#### TweetPreprocess Summary of the Pipeline and Processes

The `TweetPreprocess.ipynb` notebook outlines a comprehensive preprocessing pipeline for tweet data to prepare it for sentiment analysis and embedding generation. This data will be used in a PyTorch Geometric graph-based model to predict bill sentiment based on tweet sentiment.

### TweetPreprocess Steps in the Pipeline

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

### LawPreprocess.ipynb

#### LawPreprocess Summary of the Pipeline and Processes

The `LawPreprocess.ipynb` notebook outlines a comprehensive preprocessing pipeline for legislative bill data to prepare it for sentiment analysis and embedding generation. This data will be used in a PyTorch Geometric graph-based model to predict bill sentiment based on tweet sentiment.

### Law Preprocess Steps in the Pipeline

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

### PCAGraph.ipynb

#### PCAGraph.ipynb Summary of the Pipeline and Processes

The `PCAGraph.ipynb` notebook outlines a comprehensive pipeline for constructing a graph-based model to predict sentiment on legislative bills using tweet data. The core components include data loading, preprocessing, PCA transformation, graph construction, and building a Graph Neural Network (GNN). The pipeline explores multiple GNN architectures and performs hyperparameter tuning to identify the best-performing model.

### PCAGraph.ipynb Steps in the Pipeline

1. **Data Loading and Initial Exploration**:
   - Load tweet and bill data into DataFrames and embeddings.
   - Verify the shapes of loaded data.

2. **PCA Transformation**:
   - Apply PCA to reduce the dimensionality of tweet embeddings.
   - Ensure the dimensions of tweet and bill embeddings match.

3. **Constructing the Heterogeneous Graph**:
   - Initialize a `HeteroData` object.
   - Add bill nodes with PCA-transformed features.
   - Add tweet nodes with PCA-transformed features.
   - Establish edges between bill nodes based on state and status.

4. **Adding Tweet-Tweet Edges Based on Cosine Similarity**:
   - Calculate cosine similarity between tweet embeddings.
   - Create edges for tweet-tweet connections above a similarity threshold.

5. **Adding Tweet-Bill Edges Based on Cosine Similarity**:
   - Calculate cosine similarity between tweet and bill embeddings.
   - Create edges from tweets to their top similar bills, ensuring a manageable number of connections.

6. **Calculating Node Degree Statistics**:
   - Compute node degrees for bill-bill, tweet-tweet, and tweet-bill connections.
   - Calculate average connections and identify nodes with no connections.

7. **Adding Edge Weights**:
   - Calculate edge weights based on tweet engagement and time proximity to bill status dates.
   - Assign calculated weights to tweet-bill edges.

8. **Aggregating Sentiments**:
   - Aggregate weighted sentiments for each bill from connected tweets.
   - Save aggregated sentiments and create a tensor for bill sentiments.

9. **Updating Sentiments for Failed Bills**:
   - Adjust sentiment values for bills with failed statuses.
   - Save the updated sentiments.

10. **Splitting Data into Training, Validation, and Test Sets**:
    - Use stratified sampling to create masks for training, validation, and test sets.
    - Add these masks and the updated sentiment tensor to the graph data object.

11. **Building and Training the GNN Model**:
    - Define a GNN model using GCN layers.
    - Initialize the model, loss function, optimizer, and learning rate scheduler.
    - Train the model with early stopping based on validation loss.

12. **Evaluating the Model**:
    - Calculate evaluation metrics on validation and test sets.
    - Print the evaluation results.

13. **Hyperparameter Tuning and Experimentation**:
    - Define different GNN architectures (GCN, GAT, GraphSAGE) and hyperparameter configurations.
    - Run experiments to identify the best-performing model.

14. **Visualizing Results**:
    - Compare key metrics for top experiments.
    - Plot loss history for each experiment.

15. **Final Model Selection and Evaluation**:
    - Select and evaluate the best-performing model.
    - Plot the loss history to visualize model convergence.

## Installation

To get started with this project, you will need to clone the repository and install the required dependencies. 

### Clone the Repository

```bash
git clone https://github.com/yourusername/graph-sentiment-analysis-pipeline.git
cd graph-sentiment-analysis-pipeline
```

### Install Dependencies

We recommend using a virtual environment to manage dependencies. You can use `venv` or `conda` for this purpose.

Using `venv`:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Using `conda`:

```bash
conda create --name graph-sentiment python=3.8
conda activate graph-sentiment
pip install -r requirements.txt
```

## Usage

This section provides a brief overview of the main files and their usage.

### Jupyter Notebooks

- **PCAGraph.ipynb**: Notebook for PCA analysis and visualization, constructing the graph, and training
 the GNN model.

- **ConnectionChoices.ipynb**: Notebook for exploring different connection strategies for graph construction.
- **LawPreprocess.ipynb**: Notebook for preprocessing legislative data.
- **PipeLine.ipynb**: Main pipeline for running the entire analysis.
- **TweetPreprocess.ipynb**: Notebook for preprocessing tweet data.
- **LegiScan_APIV3.ipynb**: Notebook for fetching and processing legislative data via the LegiScan API.
- **X_APIV2.ipynb**: Notebook for fetching and processing tweet data via the Twitter API.

### Data Files

- **/TweetData/**: Contains all tweet-related data files.
- **/PCAGraph_files/**: Contains images generated during PCA analysis.
- **/Models/**: Contains model results.
- **/GraphData/**: Contains graph data files.
- **/Documents/**: Contains documentation and visualizations of the data collection pipelines and preprocessing steps.
- **/BillData/**: Contains legislative data files.

## Citations

If you use this code or the models provided in this repository, please cite the following works:

### TweetEval Sentiment Analysis Model

```bibtex
@inproceedings{barbieri-etal-2020-tweeteval,
    title = "{T}weet{E}val: Unified Benchmark and Comparative Evaluation for Tweet Classification",
    author = "Barbieri, Francesco  and
      Camacho-Collados, Jose  and
      Espinosa Anke, Luis  and
      Neves, Leonardo",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.findings-emnlp.148",
    doi = "10.18653/v1/2020.findings-emnlp.148",
    pages = "1644--1650"
}

@article{Liu2019RoBERTaAR,
  title={RoBERTa: A Robustly Optimized BERT Pretraining Approach},
  author={Yinhan Liu and Myle Ott and Naman Goyal and Jingfei Du and Mandar Joshi and Danqi Chen and Omer Levy and Mike Lewis and Luke Zettlemoyer and Veselin Stoyanov},
  journal={arXiv preprint arXiv:1907.11692},
  year={2019}
}
```

## License

This project is licensed under a dual-license model:

1. **Custom Non-Commercial License**: For non-commercial use only. See [LICENSE](./LICENSE.md) for details.
2. **Commercial License**: For commercial use, please contact Thomas Cox to obtain a commercial license.

## Contact

For any questions or commercial inquiries, please reach out to Thomas Cox at <ThomasCox273@Gmail.com>.
