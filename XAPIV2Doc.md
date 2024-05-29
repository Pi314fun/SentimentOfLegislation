# Twitter API Authentication

This script performs the following operations:

1. **Import Libraries**: Importing necessary libraries for handling Twitter API and file operations.
2. **Configuration File Handling**: It sets up a path for a configuration file (`config.json`) which contains API credentials. This path is dynamically constructed using the current directory's parent.
3. **Load Configuration**: The configuration file is opened and its JSON content (API credentials) is loaded into a variable.
4. **Twitter API Authentication**: Using the credentials from the configuration, it authenticates with Twitter using Tweepy’s `OAuthHandler`.
5. **Authentication Verification**: It attempts to verify the credentials. If successful, it prints "Authentication OK". If not, it prints the error message.

## JSON Structure Printer

This script defines and uses a function, `print_json_structure()`, to:

1. **Print JSON-Like Structure**: Recursively prints the structure of a JSON-like dictionary or list, showing the nesting and organization of data.
2. **Load and Print Data**: The function is used to load data from a specified JSON file (`tweets_data.json`) and prints its hierarchical structure.

## JSON Data Consolidation and CSV Export

This script performs the following operations:

1. **Import Libraries**: Imports the necessary libraries for handling JSON and dataframes.
2. **Functions Defined**:
    - `read_and_combine_json_files()`: Reads multiple JSON files, extracts data, handles unique identification of tweets, and combines them into a list.
    - `export_to_csv()`: Takes the combined data, converts it into a DataFrame, and exports it to a CSV file.
3. **Process and Export Data**: The functions are called with specified JSON file paths to process and export the data.

## Data Analysis with Pandas

This script utilizes Pandas to analyze and display characteristics of a dataset loaded from a CSV file:

1. **Load Dataset**: Data is loaded from `combined_tweets_data.csv`.
2. **Data Inspection and Analysis**:
    - Basic information about dataset structure and non-null count.
    - Data types of columns and unique values count.
    - Statistical summary of numeric columns.
3. **Output Descriptions**: It prints a variety of information about the dataset, including missing values, unique counts, and basic statistics.

## Observations and Recommendations:

- **Error Handling**: All scripts are well-equipped with basic error handling, particularly for file operations and data integrity issues.
- **Data Management**: The scripts effectively manage data from loading to processing and final export, making it convenient for handling large JSON datasets.
- **Code Clarity**: The code is generally well-commented, making it easier to understand its functionality. Further inline comments, especially in complex functions, could enhance readability.
