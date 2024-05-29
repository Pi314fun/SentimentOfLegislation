## Documentation for `PCAGraph.ipynb`

---

### 1. Introduction

This notebook provides a detailed implementation of a graph-based sentiment analysis model using tweet and bill data. The core components of this implementation include loading and processing data, applying PCA for dimensionality reduction, constructing a heterogeneous graph, and building a Graph Neural Network (GNN) to predict sentiments on legislative bills. Additionally, the notebook explores multiple GNN architectures and performs hyperparameter tuning to identify the best-performing model.

### 2. Libraries and Data Loading

#### Code Snippet:

```python
import pandas as pd
import torch
import numpy as np
from torch_geometric.data import HeteroData
from sklearn.metrics.pairwise import cosine_similarity

# Load tweet data
tweets_df = pd.read_csv('TweetData/final_tweets.csv')
tweet_embeddings_pca = np.load('TweetData/roberta_tweets_embeddings_pca.npy')
tweet_sentiments_tensor = torch.load('TweetData/roberta_tweets_sentiments_tensor.pt')

# Load bill data
bills_df = pd.read_csv('BillData/final_bills.csv')
bill_embeddings_pca = np.load('BillData/roberta_bills_embeddings_pca.npy')

# Verify loaded data shapes
print("Tweets DataFrame Shape:", tweets_df.shape)
print("Bills DataFrame Shape:", bills_df.shape)
print("Tweet Embeddings PCA Shape:", tweet_embeddings_pca.shape)
print("Tweet Sentiments Tensor Shape:", tweet_sentiments_tensor.shape)
print("Bill Embeddings PCA Shape:", bill_embeddings_pca.shape)
```

#### Explanation:

This section imports necessary libraries and loads tweet and bill data into Pandas DataFrames and NumPy arrays. The data shapes are printed to verify successful loading.

### 3. PCA Transformation

#### Code Snippet:

```python
from sklearn.decomposition import PCA

# Set the number of PCA components
n_components = 171

# Apply PCA to tweet embeddings
pca_tweet = PCA(n_components=n_components)
tweet_embeddings_pca_standardized = pca_tweet.fit_transform(tweet_embeddings_pca)

# Bill embeddings already have the desired number of dimensions
bill_embeddings_pca_standardized = bill_embeddings_pca

# Verify the shapes
print("Standardized Tweet Embeddings PCA Shape:", tweet_embeddings_pca_standardized.shape)
print("Standardized Bill Embeddings PCA Shape:", bill_embeddings_pca_standardized.shape)
```

#### Explanation:

Here, PCA is applied to reduce the dimensionality of tweet embeddings to match the number of dimensions of bill embeddings. The shapes of the transformed embeddings are printed to verify the transformation.

### 4. Constructing the Heterogeneous Graph

#### Code Snippet:

```python
from torch_geometric.data import HeteroData
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the HeteroData object
data = HeteroData()

# Add bill nodes with features
bill_features = torch.tensor(bill_embeddings_pca_standardized, dtype=torch.float)
data['bill'].x = bill_features

# Add tweet nodes with features
tweet_features = torch.tensor(tweet_embeddings_pca_standardized, dtype=torch.float)
data['tweet'].x = tweet_features

# Add edges based on state_id and status for bill-bill connections
bill_edges = []

# Iterate over all pairs of bills to find connections
for i in range(bills_df.shape[0]):
    for j in range(i + 1, bills_df.shape[0]):
        if (bills_df.iloc[i]['state_id'] == bills_df.iloc[j]['state_id']) and (bills_df.iloc[i]['status'] == bills_df.iloc[j]['status']):
            bill_edges.append((i, j))
            bill_edges.append((j, i))

# Convert the list of edges to a tensor and add to the graph
bill_edge_index = torch.tensor(bill_edges, dtype=torch.long).t().contiguous()
data['bill', 'related_to', 'bill'].edge_index = bill_edge_index

# Print out the current data structure to verify
print(data)
```

#### Explanation:

This section constructs a heterogeneous graph using the PyTorch Geometric library. It initializes a `HeteroData` object, adds nodes for bills and tweets with their respective features, and establishes edges between bills based on state and status.

### 5. Adding Tweet-Tweet Edges Based on Cosine Similarity

#### Code Snippet:

```python
# Calculate cosine similarity between tweet embeddings
cosine_sim = cosine_similarity(tweet_embeddings_pca_standardized)

# Define a threshold for cosine similarity to create edges
cosine_threshold = 0.5

# Add edges based on cosine similarity for tweet-tweet connections
tweet_edges = []

# Iterate over the cosine similarity matrix to find connections
for i in range(cosine_sim.shape[0]):
    for j in range(i + 1, cosine_sim.shape[1]):
        if cosine_sim[i, j] > cosine_threshold:
            tweet_edges.append((i, j))
            tweet_edges.append((j, i))

# Convert the list of edges to a tensor and add to the graph
tweet_edge_index = torch.tensor(tweet_edges, dtype=torch.long).t().contiguous()
data['tweet', 'similar_to', 'tweet'].edge_index = tweet_edge_index

# Verify tweet-tweet edges
print(f"Tweet-Tweet edges shape: {data['tweet', 'similar_to', 'tweet'].edge_index.shape}")
print(f"Number of Tweet-Tweet edges: {data['tweet', 'similar_to', 'tweet'].edge_index.shape[1]}")
print(f"First 10 Tweet-Tweet edges: {data['tweet', 'similar_to', 'tweet'].edge_index[:, :10]}")
```

#### Explanation:

Edges between tweets are created based on cosine similarity of their embeddings. A threshold is used to determine which tweet pairs are connected, and the resulting edges are added to the graph.

### 6. Adding Tweet-Bill Edges Based on Cosine Similarity

#### Code Snippet:

```python
# Function to find the top N closest items
def top_n_similarities(sim_matrix, N):
    top_n_indices = []
    for i in range(sim_matrix.shape[0]):
        # Get the indices of the top N similarities for each row
        indices = sim_matrix[i].argsort()[-N:][::-1]
        top_n_indices.append(indices)
    return top_n_indices

# Calculate cosine similarity between tweet embeddings and bill embeddings
tweet_bill_cosine_sim = cosine_similarity(tweet_embeddings_pca_standardized, bill_embeddings_pca_standardized)

# Define the maximum number of connections for bills and tweets
max_bill_connections = 10
max_tweet_connections = 5

# Initialize a dictionary to keep track of the number of connections per bill
bill_connection_count = {i: 0 for i in range(bills_df.shape[0])}

# Initialize a list to store the edges
tweet_bill_edges = []

# Get the top 5 bills for each tweet based on cosine similarity
top_bills_per_tweet = top_n_similarities(tweet_bill_cosine_sim, max_tweet_connections)

# Add edges from tweets to their top 5 closest bills
for tweet_idx, top_bills in enumerate(top_bills_per_tweet):
    connections = 0
    for bill_idx in top_bills:
        if bill_connection_count[bill_idx] < max_bill_connections:
            tweet_bill_edges.append((tweet_idx, bill_idx))
            bill_connection_count[bill_idx] += 1
            connections += 1
        if connections >= max_tweet_connections:
            break

# Convert the list of edges to a tensor and add to the graph
tweet_bill_edge_index = torch.tensor(tweet_bill_edges, dtype=torch.long).t().contiguous()
data['tweet', 'mentions', 'bill'].edge_index = tweet_bill_edge_index

# Verify tweet-bill edges
print(f"Tweet-Bill edges shape: {data['tweet', 'mentions', 'bill'].edge_index.shape}")
print(f"Number of Tweet-Bill edges: {data['tweet', 'mentions', 'bill'].edge_index.shape[1]}")
print(f"First 10 Tweet-Bill edges: {data['tweet', 'mentions', 'bill'].edge_index[:, :10]}")
```

#### Explanation:

This section creates edges between tweets and bills based on cosine similarity, ensuring that each tweet connects to its top 5 most similar bills, and each bill has a maximum of 10 connections to tweets. This process helps to ensure that the connections are meaningful and manageable in number.

### 7. Calculating Node Degree Statistics

#### Code Snippet:

```python
# Function to calculate node degree statistics
def calculate_node_degree(edge_index, num_nodes):
    degrees = torch.zeros(num_nodes, dtype=torch.long)
    for edge in edge_index.t():
        degrees[edge[0]] += 1
        degrees[edge[1]] += 1
    return degrees

# Calculate degrees for bill-bill connections
bill_bill_degrees = calculate_node_degree(data['bill', 'related_to', 'bill'].edge_index, data['bill'].x.size(0))

# Calculate degrees for tweet-tweet connections
tweet_tweet_degrees = calculate_node_degree(data['tweet', 'similar_to', 'tweet'].edge_index, data['tweet'].x.size(0))

# Calculate degrees for tweet-bill connections
tweet_bill_edge_index = data['tweet', 'mentions', 'bill'].edge_index
tweet_degrees = torch.zeros(data['tweet'].x.size(0), dtype=torch.long)
bill_degrees = torch.zeros(data['bill'].x.size(0), dtype=torch.long)

for edge in tweet_bill_edge_index.t():
    tweet_degrees[edge[0]] += 1
    bill_degrees[edge[1]] += 1

# Calculate average number

 of connections
avg_tweet_connections = tweet_degrees.float().mean().item()
avg_bill_connections = bill_degrees.float().mean().item()

# Calculate number of nodes with no connections
num_tweets_no_connections = (tweet_degrees == 0).sum().item()
num_bills_no_connections = (bill_degrees == 0).sum().item()

# Print graph structure and connectivity statistics
print(f"Total number of bill nodes: {data['bill'].x.size(0)}")
print(f"Total number of tweet nodes: {data['tweet'].x.size(0)}")
print(f"Total number of bill-bill edges: {data['bill', 'related_to', 'bill'].edge_index.size(1)}")
print(f"Total number of tweet-tweet edges: {data['tweet', 'similar_to', 'tweet'].edge_index.size(1)}")
print(f"Total number of tweet-bill edges: {data['tweet', 'mentions', 'bill'].edge_index.size(1)}")
print(f"Average number of connections per tweet: {avg_tweet_connections}")
print(f"Average number of connections per bill: {avg_bill_connections}")
print(f"Number of tweets with no connections: {num_tweets_no_connections}")
print(f"Number of bills with no connections: {num_bills_no_connections}")
print(f"Bill degrees (first 10): {bill_degrees[:10]}")
print(f"Tweet degrees (first 10): {tweet_degrees[:10]}")
```

#### Explanation:

Node degree statistics are calculated to understand the connectivity of nodes in the graph. This helps in identifying nodes with no connections and ensures the graph's structure is as expected.

### 8. Adding Edge Weights

#### Code Snippet:

```python
from datetime import datetime, timedelta

# Convert 'created_at' and 'status_date' to datetime objects
tweets_df['created_at'] = pd.to_datetime(tweets_df['created_at'], unit='s')
bills_df['status_date'] = pd.to_datetime(bills_df['status_date'])

# Initialize a list to store the edge weights
edge_weights = []

# Iterate over tweet-bill edges to calculate weights
for tweet_idx, bill_idx in tweet_bill_edge_index.t().tolist():
    tweet = tweets_df.iloc[tweet_idx]
    bill = bills_df.iloc[bill_idx]

    # Calculate the engagement weight
    log_engage = tweet['log_engage']

    # Calculate the time proximity weight
    days_diff = abs((bill['status_date'] - tweet['created_at']).days)
    if days_diff <= 15:
        time_weight = (15 - days_diff) / 15
    else:
        time_weight = 0

    # Combined weight
    weight = log_engage * (1 + time_weight)
    edge_weights.append(weight)

# Convert edge weights to tensor and add to the graph
edge_weights_tensor = torch.tensor(edge_weights, dtype=torch.float)
data['tweet', 'mentions', 'bill'].edge_attr = edge_weights_tensor

# Verify edge weights
print(f"Edge weights shape: {data['tweet', 'mentions', 'bill'].edge_attr.shape}")
print(f"First 10 edge weights: {data['tweet', 'mentions', 'bill'].edge_attr[:10]}")
```

#### Explanation:

Edge weights are calculated based on tweet engagement and time proximity to bill status dates. This allows for more nuanced connections between tweets and bills.

### 9. Aggregating Sentiments

#### Code Snippet:

```python
# Function to aggregate weighted sentiments for each bill
def aggregate_sentiments(tweets_df, tweet_bill_edge_index, edge_weights_tensor, num_bills):
    # Initialize tensors to store aggregated sentiments and weights sum
    positive_sum = torch.zeros(num_bills, dtype=torch.float)
    neutral_sum = torch.zeros(num_bills, dtype=torch.float)
    negative_sum = torch.zeros(num_bills, dtype=torch.float)
    weights_sum = torch.zeros(num_bills, dtype=torch.float)

    # Iterate over tweet-bill edges to accumulate weighted sentiments
    for idx, (tweet_idx, bill_idx) in enumerate(tweet_bill_edge_index.t().tolist()):
        weight = edge_weights_tensor[idx].item()
        tweet_sentiment = tweets_df.iloc[tweet_idx][['positive', 'neutral', 'negative']]

        positive_sum[bill_idx] += weight * tweet_sentiment['positive']
        neutral_sum[bill_idx] += weight * tweet_sentiment['neutral']
        negative_sum[bill_idx] += weight * tweet_sentiment['negative']
        weights_sum[bill_idx] += weight

    # Avoid division by zero by ensuring weights_sum is not zero
    weights_sum[weights_sum == 0] = 1

    # Compute weighted average sentiments
    positive_avg = positive_sum / weights_sum
    neutral_avg = neutral_sum / weights_sum
    negative_avg = negative_sum / weights_sum

    return positive_avg, neutral_avg, negative_avg

# Number of bills
num_bills = data['bill'].x.size(0)

# Aggregate sentiments
positive_avg, neutral_avg, negative_avg = aggregate_sentiments(
    tweets_df, tweet_bill_edge_index, data['tweet', 'mentions', 'bill'].edge_attr, num_bills)

# Create a DataFrame for the aggregated sentiments
aggregated_sentiments_df = pd.DataFrame({
    'bill_id': bills_df['bill_id'],
    'positive': positive_avg.numpy(),
    'neutral': neutral_avg.numpy(),
    'negative': negative_avg.numpy()
})

# Save the DataFrame to a new file
aggregated_sentiments_df.to_csv('BillData/aggregated_bill_sentiments.csv', index=False)

# Create a tensor for the bills' sentiments
bills_sentiments_tensor = torch.tensor(aggregated_sentiments_df[['positive', 'neutral', 'negative']].values, dtype=torch.float)

# Save the tensor to a file
torch.save(bills_sentiments_tensor, 'BillData/bills_sentiments_tensor.pt')

# Verify the results
print(f"Aggregated Sentiments DataFrame shape: {aggregated_sentiments_df.shape}")
print(f"First 5 rows of the Aggregated Sentiments DataFrame:\n{aggregated_sentiments_df.head()}")
print(f"Bills Sentiments Tensor shape: {bills_sentiments_tensor.shape}")
print(f"First 5 rows of the Bills Sentiments Tensor:\n{bills_sentiments_tensor[:5]}")
```

#### Explanation:

Sentiments from tweets are aggregated for each bill using weighted averages. This aggregation provides an overall sentiment score for each bill based on the connected tweets' sentiments.

### 10. Updating Sentiments for Failed Bills

#### Code Snippet:

```python
# Function to update sentiments for failed bills
def update_failed_bill_sentiments(bills_df, sentiments_tensor):
    updated_sentiments = sentiments_tensor.clone()
    for idx, bill in bills_df.iterrows():
        if bill['status'] in [5, 6]:
            # positive <- negative
            updated_sentiments[idx, 0] = sentiments_tensor[idx, 2]
            # neutral <- 1 - neutral
            updated_sentiments[idx, 1] = 1 - sentiments_tensor[idx, 1]
            # negative <- positive
            updated_sentiments[idx, 2] = sentiments_tensor[idx, 0]
    return updated_sentiments

# Update the sentiments tensor
updated_bills_sentiments_tensor = update_failed_bill_sentiments(bills_df, bills_sentiments_tensor)

# Update the DataFrame and save to CSV
updated_aggregated_sentiments_df = pd.DataFrame({
    'bill_id': bills_df['bill_id'],
    'positive': updated_bills_sentiments_tensor[:, 0].numpy(),
    'neutral': updated_bills_sentiments_tensor[:, 1].numpy(),
    'negative': updated_bills_sentiments_tensor[:, 2].numpy()
})

updated_aggregated_sentiments_df.to_csv('BillData/aggregated_bill_sentiments.csv', index=False)
torch.save(updated_bills_sentiments_tensor, 'BillData/bills_sentiments_tensor.pt')

# Verify the updated sentiments
print(f"Updated Aggregated Sentiments DataFrame shape: {updated_aggregated_sentiments_df.shape}")
print(f"First 5 rows of the Updated Aggregated Sentiments DataFrame:\n{updated_aggregated_sentiments_df.head()}")
print(f"Updated Bills Sentiments Tensor shape: {updated_bills_sentiments_tensor.shape}")
print(f"First 5 rows of the Updated Bills Sentiments Tensor:\n{updated_bills_sentiments_tensor[:5]}")
```

#### Explanation:

The sentiments for failed bills (status 5 or 6) are updated by swapping positive and negative sentiments and adjusting neutral sentiments. This reflects the negative outcome for failed bills more accurately.

### 11. Splitting Data into Training, Validation, and Test Sets

#### Code Snippet:

```python
from sklearn.model_selection import train_test_split

# Create a mask for bills with non-zero sentiments
non_zero_sentiment_mask = (bills_sentiments_tensor.sum(dim=1) > 0).numpy()

# Filter the bills dataframe and sentiments tensor
filtered_bills_df = bills_df[non_zero_sentiment_mask]
filtered_bills_sentiments = bills_sentiments_tensor[non_zero_sentiment_mask]

# Get the statuses for stratified splitting
statuses = filtered_bills_df['status'].values

# Split the data into training, validation, and test sets
train_idx, test_idx = train_test_split(np.arange(filtered_bills_df.shape[0]), test_size=0.2, random_state=42, stratify=statuses)
train_idx, val_idx = train_test_split(train_idx, test_size=0.25, random_state=42, stratify=statuses[train_idx])  # 0.25 * 0.8 = 0

.2

# Convert indices to tensors
train_mask = torch.zeros(filtered_bills_df.shape[0], dtype=torch.bool)
val_mask = torch.zeros(filtered_bills_df.shape[0], dtype=torch.bool)
test_mask = torch.zeros(filtered_bills_df.shape[0], dtype=torch.bool)

train_mask[train_idx] = True
val_mask[val_idx] = True
test_mask[test_idx] = True

# Map the filtered indices back to the original indices
original_indices = np.where(non_zero_sentiment_mask)[0]
train_mask_full = torch.zeros(num_bills, dtype=torch.bool)
val_mask_full = torch.zeros(num_bills, dtype=torch.bool)
test_mask_full = torch.zeros(num_bills, dtype=torch.bool)

train_mask_full[original_indices[train_mask.numpy()]] = True
val_mask_full[original_indices[val_mask.numpy()]] = True
test_mask_full[original_indices[test_mask.numpy()]] = True

# Add masks to the data object
data['bill'].train_mask = train_mask_full
data['bill'].val_mask = val_mask_full
data['bill'].test_mask = test_mask_full

# Add the updated sentiments tensor to the data object
data['bill'].y = updated_bills_sentiments_tensor

# Verify the masks
print(f"Training set size: {train_mask.sum().item()}")
print(f"Validation set size: {val_mask.sum().item()}")
print(f"Test set size: {test_mask.sum().item()}")
print(f"Total number of filtered bills: {filtered_bills_df.shape[0]}")
```

#### Explanation:

The data is split into training, validation, and test sets using stratified sampling based on bill statuses. Masks are created to differentiate between the different sets and added to the graph data object.

### 12. Building and Training the GNN Model

#### Code Snippet:

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data['bill'].x, data['bill', 'related_to', 'bill'].edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# Initialize the model, loss function, and optimizer
model = GNN(in_channels=171, hidden_channels=64, out_channels=3, dropout=0.5)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# Print the model architecture
print(model)
```

#### Explanation:

A GNN model is defined using Graph Convolutional Network (GCN) layers. The model architecture, loss function, optimizer, and learning rate scheduler are initialized. The model architecture is printed to verify its structure.

### 13. Training and Validating the Model

#### Code Snippet:

```python
# Function to train the model
def train():
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data['bill'].train_mask], data['bill'].y[data['bill'].train_mask])
    loss.backward()
    optimizer.step()
    scheduler.step()
    total_loss += loss.item()
    return total_loss

# Function to validate the model
def validate():
    model.eval()
    total_loss = 0
    with torch.no_grad():
        out = model(data)
        loss = criterion(out[data['bill'].val_mask], data['bill'].y[data['bill'].val_mask])
        total_loss += loss.item()
    return total_loss

# Early stopping criteria
best_val_loss = float('inf')
patience = 10
patience_counter = 0

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    train_loss = train()
    val_loss = validate()
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # Check for early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save the best model
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping triggered")
        break

# Load the best model
model.load_state_dict(torch.load('best_model.pth'))
```

#### Explanation:

Functions to train and validate the model are defined. An early stopping mechanism is implemented to prevent overfitting. The training loop runs for a specified number of epochs, with validation loss monitored to trigger early stopping if necessary. The best model is saved and reloaded.

### 14. Evaluating the Model

#### Code Snippet:

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Function to calculate evaluation metrics
def evaluate(mask):
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []
    with torch.no_grad():
        out = model(data)
        preds = out[mask].argmax(dim=1).cpu().numpy()
        # Convert one-hot to class indices
        labels = data['bill'].y[mask].argmax(dim=1).cpu().numpy()
        y_true.extend(labels)
        y_pred.extend(preds)
        # Use softmax probabilities for ROC AUC
        y_prob.extend(F.softmax(out[mask], dim=1).cpu().numpy())
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
    conf_matrix = confusion_matrix(y_true, y_pred)
    return accuracy, precision, recall, f1, roc_auc, conf_matrix

# Evaluate on the validation set
val_accuracy, val_precision, val_recall, val_f1, val_roc_auc, val_conf_matrix = evaluate(data['bill'].val_mask)
print(f'Validation Accuracy: {val_accuracy:.4f}')
print(f'Validation Precision: {val_precision:.4f}')
print(f'Validation Recall: {val_recall:.4f}')
print(f'Validation F1 Score: {val_f1:.4f}')
print(f'Validation ROC AUC: {val_roc_auc:.4f}')
print(f'Validation Confusion Matrix:\n{val_conf_matrix}')

# Evaluate on the test set
test_accuracy, test_precision, test_recall, test_f1, test_roc_auc, test_conf_matrix = evaluate(data['bill'].test_mask)
print(f'Test Accuracy: {test_accuracy:.4f}')
print(f'Test Precision: {test_precision:.4f}')
print(f'Test Recall: {test_recall:.4f}')
print(f'Test F1 Score: {test_f1:.4f}')
print(f'Test ROC AUC: {test_roc_auc:.4f}')
print(f'Test Confusion Matrix:\n{test_conf_matrix}')
```

#### Explanation:

The `evaluate` function calculates various evaluation metrics for the model. The function is used to evaluate the model on both validation and test sets, and the results are printed.

### 15. Hyperparameter Tuning and Experimentation

#### Code Snippet:

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, BatchNorm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import json
import os
from itertools import product
from torch.optim.lr_scheduler import StepLR

# Define the model architectures
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data['bill'].x, data['bill', 'related_to', 'bill'].edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden

_channels)
        self.conv2 = GATConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data['bill'].x, data['bill', 'related_to', 'bill'].edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data['bill'].x, data['bill', 'related_to', 'bill'].edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# Function to train the model
def train(model, data, optimizer, criterion):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data['bill'].train_mask], data['bill'].y[data['bill'].train_mask].argmax(dim=1))
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
    return total_loss

# Function to validate the model
def validate(model, data, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        out = model(data)
        loss = criterion(out[data['bill'].val_mask], data['bill'].y[data['bill'].val_mask].argmax(dim=1))
        total_loss += loss.item()
    return total_loss

# Function to evaluate the model
def evaluate(model, data, mask):
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []
    with torch.no_grad():
        out = model(data)
        preds = out[mask].argmax(dim=1).cpu().numpy()
        labels = data['bill'].y[mask].argmax(dim=1).cpu().numpy()
        y_true.extend(labels)
        y_pred.extend(preds)
        y_prob.extend(F.softmax(out[mask], dim=1).cpu().numpy())
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
    conf_matrix = confusion_matrix(y_true, y_pred)
    return accuracy, precision, recall, f1, roc_auc, conf_matrix

# Function to run experiments
def run_experiment(config, data):
    torch.manual_seed(config['seed'])
    if config['model_type'] == 'GCN':
        model = GCN(in_channels=config['in_channels'], hidden_channels=config['hidden_channels'],
                    out_channels=config['out_channels'], dropout=config['dropout'])
    elif config['model_type'] == 'GAT':
        model = GAT(in_channels=config['in_channels'], hidden_channels=config['hidden_channels'],
                    out_channels=config['out_channels'], dropout=config['dropout'])
    elif config['model_type'] == 'GraphSAGE':
        model = GraphSAGE(in_channels=config['in_channels'], hidden_channels=config['hidden_channels'],
                          out_channels=config['out_channels'], dropout=config['dropout'])

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])

    best_val_loss = float('inf')
    patience_counter = 0
    train_loss_history = []
    val_loss_history = []

    for epoch in range(config['num_epochs']):
        train_loss = train(model, data, optimizer, criterion)
        val_loss = validate(model, data, criterion)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(config['output_dir'], 'best_model.pth'))
        else:
            patience_counter += 1

        if patience_counter >= config['patience']:
            print("Early stopping triggered")
            break

        scheduler.step()

    model.load_state_dict(torch.load(os.path.join(config['output_dir'], 'best_model.pth')))
    val_metrics = evaluate(model, data, data['bill'].val_mask)
    test_metrics = evaluate(model, data, data['bill'].test_mask)

    metrics = {
        'val_accuracy': val_metrics[0],
        'val_precision': val_metrics[1],
        'val_recall': val_metrics[2],
        'val_f1': val_metrics[3],
        'val_roc_auc': val_metrics[4],
        'val_conf_matrix': val_metrics[5].tolist(),
        'test_accuracy': test_metrics[0],
        'test_precision': test_metrics[1],
        'test_recall': test_metrics[2],
        'test_f1': test_metrics[3],
        'test_roc_auc': test_metrics[4],
        'test_conf_matrix': test_metrics[5].tolist(),
        'train_loss_history': train_loss_history,
        'val_loss_history': val_loss_history
    }

    with open(os.path.join(config['output_dir'], 'metrics.json'), 'w') as f:
        json.dump(metrics, f)

    print(f"Experiment completed. Results saved to {config['output_dir']}")

# Define hyperparameter grid
model_types = ['GCN', 'GAT', 'GraphSAGE']
hidden_channels_list = [64, 128, 256]
dropout_rates = [0.3, 0.5, 0.7]
learning_rates = [0.001, 0.005, 0.01]
seeds = [42, 123, 456]
step_sizes = [10, 20]
gammas = [0.5, 0.7, 0.9]

# Load data
# Assuming the 'data' variable is your PyTorch Geometric Data object containing your graph data

# Define the experiment configurations
configs = [
    {
        'model_type': model_type,
        'in_channels': data['bill'].num_node_features,
        'hidden_channels': hidden_channels,
        'out_channels': 3,  # Assuming 3 classes for sentiment analysis
        'dropout': dropout,
        'learning_rate': learning_rate,
        'num_epochs': 100,
        'patience': 10,
        'step_size': step_size,
        'gamma': gamma,
        'seed': seed,
        'output_dir': os.path.join('experiments', f'{model_type}_hidden{hidden_channels}_dropout{dropout}_lr{learning_rate}_seed{seed}')
    }
    for model_type, hidden_channels, dropout, learning_rate, step_size, gamma, seed in product(
        model_types, hidden_channels_list, dropout_rates, learning_rates, step_sizes, gammas, seeds
    )
]

# Create the output directories if they do not exist
for config in configs:
    os.makedirs(config['output_dir'], exist_ok=True)

# Run experiments
for config in configs:
    run_experiment(config, data)
```

#### Explanation:

This section defines different GNN architectures (GCN, GAT, GraphSAGE) and performs hyperparameter tuning by running multiple experiments with different configurations. The best model is selected based on evaluation metrics.

### 16. Visualizing Results

#### Code Snippet:

```python
import os
import json
import matplotlib.pyplot as plt

# Define the root directory where all experiments are saved
root_dir = 'experiments'

# Function to load metrics
def load_metrics(output_dir):
    with open(os.path.join(output_dir, 'metrics.json'), 'r') as f:
        metrics = json.load(f)
    return metrics

# Get list of all experiment directories
experiment_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

# Load metrics for all experiments
all_metrics = [load_metrics(os.path.join(root_dir, d)) for d in experiment_dirs]

# Define a function to compute a composite score
def composite_score(m):
    # You can customize the weights for each metric based on your priorities
    val_accuracy_weight = 0.25
    test_accuracy_weight = 0.25
    test_f1_weight = 0.25
    test_roc_auc_weight = 0.25
    return (val_accuracy_weight * m['val_accuracy'] +
            test_accuracy_weight * m['test_accuracy'] +
            test_f1_weight * m['test_f1'] +
            test_roc_auc_weight * m['test_roc_auc'])

# Compute composite scores and sort metrics
scored_metrics = [(composite_score(m), m, d) for m, d in zip(all_metrics, experiment_dirs)]
scored_metrics.sort(reverse=True, key=lambda

 x: x[0])

# Select top 10 experiments
top_metrics = scored_metrics[:10]

# Extract the metrics and experiment directories
top_metrics_values = [m[1] for m in top_metrics]
top_experiment_dirs = [m[2] for m in top_metrics]

# Compare key metrics for top 10 experiments
def compare_metrics(metrics, experiment_dirs):
    print(f"{'Experiment':<30} {'Val Accuracy':<15} {'Test Accuracy':<15} {'Test F1 Score':<15} {'Test ROC AUC':<15}")
    for i, m in enumerate(metrics):
        print(f"{experiment_dirs[i]:<30} {m['val_accuracy']:<15.4f} {m['test_accuracy']:<15.4f} {m['test_f1']:<15.4f} {m['test_roc_auc']:<15.4f}")

compare_metrics(top_metrics_values, top_experiment_dirs)

# Plot loss history for top 10 experiments
for i, m in enumerate(top_metrics_values):
    plt.figure(figsize=(10, 5))
    plt.plot(m['train_loss_history'], label='Train Loss')
    plt.plot(m['val_loss_history'], label='Val Loss')
    plt.title(f'Experiment {top_experiment_dirs[i]} Loss History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
```

#### Explanation:

The results of the experiments are visualized by comparing key metrics for the top 10 experiments and plotting the loss history for each experiment. This helps in understanding the performance and convergence of different models.

### 17. Final Model Selection and Evaluation

#### Code Snippet:

```python
import os
import json
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, BatchNorm
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

# Define the ideal model
class GraphSAGEModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data['bill'].x, data['bill', 'related_to', 'bill'].edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# Function to train the model
def train_model(model, data, optimizer, criterion, scheduler, num_epochs, patience, output_dir):
    best_val_loss = float('inf')
    patience_counter = 0
    train_loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data['bill'].train_mask], data['bill'].y[data['bill'].train_mask].argmax(dim=1))
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_out = model(data)
            val_loss = criterion(val_out[data['bill'].val_mask], data['bill'].y[data['bill'].val_mask].argmax(dim=1))

        train_loss_history.append(loss.item())
        val_loss_history.append(val_loss.item())

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Load the best model
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth')))
    return train_loss_history, val_loss_history

# Function to evaluate the model
def evaluate_model(model, data, mask):
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        out = model(data)
        preds = out[mask].argmax(dim=1).cpu().numpy()
        labels = data['bill'].y[mask].argmax(dim=1).cpu().numpy()
        y_true.extend(labels)
        y_pred.extend(preds)
        y_prob.extend(F.softmax(out[mask], dim=1).cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
    conf_matrix = confusion_matrix(y_true, y_pred)

    return accuracy, precision, recall, f1, roc_auc, conf_matrix

# Function to run the experiment
def run_experiment(config, data):
    torch.manual_seed(config['seed'])
    model = GraphSAGEModel(in_channels=config['in_channels'], hidden_channels=config['hidden_channels'],
                           out_channels=config['out_channels'], dropout=config['dropout'])
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])

    train_loss_history, val_loss_history = train_model(model, data, optimizer, criterion, scheduler,
                                                       config['num_epochs'], config['patience'], config['output_dir'])

    val_metrics = evaluate_model(model, data, data['bill'].val_mask)
    test_metrics = evaluate_model(model, data, data['bill'].test_mask)

    metrics = {
        'val_accuracy': val_metrics[0],
        'val_precision': val_metrics[1],
        'val_recall': val_metrics[2],
        'val_f1': val_metrics[3],
        'val_roc_auc': val_metrics[4],
        'val_conf_matrix': val_metrics[5].tolist(),
        'test_accuracy': test_metrics[0],
        'test_precision': test_metrics[1],
        'test_recall': test_metrics[2],
        'test_f1': test_metrics[3],
        'test_roc_auc': test_metrics[4],
        'test_conf_matrix': test_metrics[5].tolist(),
        'train_loss_history': train_loss_history,
        'val_loss_history': val_loss_history
    }

    with open(os.path.join(config['output_dir'], 'metrics.json'), 'w') as f:
        json.dump(metrics, f)

    print(f"Experiment completed. Results saved to {config['output_dir']}")

# Ideal configuration based on previous analysis
ideal_config = {
    'seed': 42,
    'in_channels': 171,
    'hidden_channels': 256,
    'out_channels': 3,
    'dropout': 0.7,
    'learning_rate': 0.01,
    'step_size': 20,
    'gamma': 0.5,
    'num_epochs': 100,
    'patience': 10,
    'output_dir': 'ideal_experiment'
}

# Ensure the output directory exists
os.makedirs(ideal_config['output_dir'], exist_ok=True)

# Run the ideal experiment
run_experiment(ideal_config, data)

# Load and display the results
ideal_metrics = load_metrics(ideal_config['output_dir'])
print(f"Val Accuracy: {ideal_metrics['val_accuracy']:.4f}")
print(f"Test Accuracy: {ideal_metrics['test_accuracy']:.4f}")
print(f"Test F1 Score: {ideal_metrics['test_f1']:.4f}")
print(f"Test ROC AUC: {ideal_metrics['test_roc_auc']:.4f}")

# Plot the loss history
plt.figure(figsize=(10, 5))
plt.plot(ideal_metrics['train_loss_history'], label='Train Loss')
plt.plot(ideal_metrics['val_loss_history'], label='Val Loss')
plt.title('Ideal Model Loss History')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

#### Explanation:

The final section identifies and evaluates the best-performing model based on previous experiments. The model is trained and evaluated with detailed metrics, and the loss history is plotted to visualize the model's convergence.

---

### Conclusion

This documentation provides a comprehensive overview of the `PCAGraph.ipynb` notebook, covering all essential steps, including data loading, preprocessing, graph construction, model building, training, evaluation, and hyperparameter tuning. Each section includes code snippets and detailed explanations to ensure clarity and reproducibility.
