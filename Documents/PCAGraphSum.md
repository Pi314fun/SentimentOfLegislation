# Summary of the Pipeline and Processes

## Overview

The `PCAGraph.ipynb` notebook outlines a comprehensive pipeline for constructing a graph-based model to predict sentiment on legislative bills using tweet data. The core components include data loading, preprocessing, PCA transformation, graph construction, and building a Graph Neural Network (GNN). The pipeline explores multiple GNN architectures and performs hyperparameter tuning to identify the best-performing model.

### Steps in the Pipeline

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

### Conclusion

This pipeline constructs a sophisticated graph-based model to predict sentiments on legislative bills based on tweet data. Each step, from data loading and PCA transformation to model training and evaluation, ensures a robust and accurate sentiment prediction model using PyTorch Geometric. The comprehensive experimentation and visualization facilitate identifying the best-performing model, ensuring high reliability in sentiment analysis.
