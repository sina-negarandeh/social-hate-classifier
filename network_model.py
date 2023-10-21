# network_model.py
# This module contains the code for embedding the social structure using GraphSAGE

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn.models import GraphSAGE
from torch_geometric.utils import negative_sampling

# Load the train and test matrix data frames from csv files
train_matrix_df = pd.read_csv('train_matrix_df.csv')
test_matrix_df = pd.read_csv('test_matrix_df.csv')

# Create the source and target nodes lists for the train set
source_nodes = []
target_nodes = []
for i, row in train_matrix_df.iterrows():
    source_nodes.extend([row["index"]] * len(row["follower_index"]))
    target_nodes.extend(row["follower_index"])

# Create the node features and edge index tensors for the train set
x = torch.tensor(train_matrix_df[["num_followers", "num_texts"]].values, dtype=torch.float32)
edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
data = Data(x=x, edge_index=edge_index)
data.x = torch.nan_to_num(data.x)

# Load the GraphSAGE model from PyTorch Geometric
model = GraphSAGE(in_channels=data.num_features, hidden_channels=64, num_layers=3, out_channels=16)

# Define the optimizer and learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.00001)

# Define a contrastive loss function to learn the embeddings from positive and negative edges
def contrastive_loss(embeddings, edge_index):
    num_nodes = embeddings.size(0)
    num_edges = edge_index.size(1)

    embeddings = F.normalize(embeddings, p=2.0)
    pos_score = torch.sum(embeddings[edge_index[0]] * embeddings[edge_index[1]], dim=-1)
    neg_edge_index = negative_sampling(edge_index=edge_index,
                                       num_nodes=num_nodes,
                                       num_neg_samples=num_edges)
    neg_score = torch.sum(embeddings[neg_edge_index[0]] * embeddings[neg_edge_index[1]], dim=-1)
    
    epsilon = 1e-15
    loss = -torch.log(torch.sigmoid(pos_score) + epsilon).mean()
    loss += -torch.log(1 - torch.sigmoid(neg_score) + epsilon).mean()
    return loss

# Train the model on the train set for 500 epochs
for epoch in range(500):
    model.train()
    optimizer.zero_grad()
    embeddings = model(data.x, data.edge_index)
    loss = contrastive_loss(embeddings, data.edge_index)

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item():.4f}")

# Get the embeddings for the train set and save them as a new column in the train matrix data frame
model.eval()
with torch.no_grad():
 embeddings = model(data.x, data.edge_index)

features_list = embeddings.tolist()
train_matrix_df['graph_embeddings'] = features_list

# Create the source and target nodes lists for the test set
source_nodes = []
target_nodes = []
for i, row in test_matrix_df.iterrows():
    source_nodes.extend([row["index"]] * len(row["follower_index"]))
    target_nodes.extend(row["follower_index"])

# Create the node features and edge index tensors for the test set
x = torch.tensor(test_matrix_df[["num_followers", "num_texts"]].values, dtype=torch.float32)
edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
data = Data(x=x, edge_index=edge_index)
data.x = torch.nan_to_num(data.x)

# Get the embeddings for the test set and save them as a new column in the test matrix data frame
model.eval()
with torch.no_grad():
 embeddings = model(data.x, data.edge_index)

features_list = embeddings.tolist()
test_matrix_df['graph_embeddings'] = features_list
