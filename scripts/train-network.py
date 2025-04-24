import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import pickle
import networkx as nx
import scipy.sparse as sp
import os


# Set the path from which you want to load the graph
input_pickle = "network.gpickle"

# Load the graph from the pickle file
with open(input_pickle, "rb") as f:
    G_loaded = pickle.load(f)

print("Graph reloaded from pickle.")
G = G_loaded

# -------------------------------
# Generate Node Features
# -------------------------------
# We follow the dimensions from your methods:
#   - Chemical (Drug): 1224-d (1024-d ECFP6 + 200-d LSTM)
#   - Gene (Protein): 1280-d
#   - Variant: 128-d
#   - Disease: 512-d
# For any unknown type, we default to a 100-d vector.
node_features = {}
for node, data in G.nodes(data=True):
    ntype = data.get("type", "Unknown")
    if ntype == "Chemical":
        feat = np.random.rand(1224)
    elif ntype == "Gene":
        feat = np.random.rand(1280)
    elif ntype == "Variant":
        feat = np.random.rand(128)
    elif ntype == "Disease":
        feat = np.random.rand(512)
    else:
        feat = np.random.rand(100)
    node_features[node] = feat

# Helper to pad feature vectors to the target dimension (1280)
def pad_features(vec, target_dim=1280):
    current_dim = vec.shape[0]
    if current_dim < target_dim:
        return np.pad(vec, (0, target_dim - current_dim), mode='constant')
    else:
        return vec[:target_dim]

# Create an ordered node list and a mapping to indices
node_list = list(G.nodes())
node_idx = {node: i for i, node in enumerate(node_list)}
X_list = [pad_features(node_features[node], target_dim=1280) for node in node_list]
X = torch.tensor(np.vstack(X_list), dtype=torch.float32)

# -------------------------------
# Precompute Adjacency Matrices for SIGN (Sparse Version)
# -------------------------------
def preprocess_adj_sparse(A, K):
    """
    A: a scipy.sparse CSR adjacency matrix.
    K: number of hops.

    Computes the normalized adjacency matrix A_norm and then its powers A_norm^k for k = 0...K.
    Returns a list of torch sparse tensors.
    """
    I = sp.eye(A.shape[0], format='csr')
    A_tilde = A + I

    d = np.array(A_tilde.sum(axis=1)).flatten()
    d_inv_sqrt = 1.0 / np.sqrt(d)
    D_inv_sqrt = sp.diags(d_inv_sqrt)

    A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt

    As_list = []
    current_A = sp.eye(A.shape[0], format='csr')  # A_norm^0
    for k in range(K+1):
        if k > 0:
            current_A = current_A.dot(A_norm)
        current_A_coo = current_A.tocoo()
        indices = torch.from_numpy(
            np.vstack((current_A_coo.row, current_A_coo.col)).astype(np.int64)
        )
        values = torch.from_numpy(current_A_coo.data.astype(np.float32))
        shape = torch.Size(current_A_coo.shape)
        sparse_tensor = torch.sparse_coo_tensor(
            indices, values, shape, dtype=torch.float32, device=values.device
        )
        As_list.append(sparse_tensor)

    return As_list

# File where we store A_sparse and K
gpickle_path = 'A_sparse_and_K.gpickle'

if os.path.exists(gpickle_path):
    # Load existing adjacency and K
    with open(gpickle_path, 'rb') as f:
        A_sparse, K = pickle.load(f)
    print(f"Loaded A_sparse and K={K} from '{gpickle_path}'")
else:
    # Build and save adjacency + K
    # (Assuming G and node_list are already defined)
    A_sparse = nx.to_scipy_sparse_array(G, nodelist=node_list, weight=None, format='csr')
    K = 3
    with open(gpickle_path, 'wb') as f:
        pickle.dump((A_sparse, K), f)
    print(f"Built and saved A_sparse and K={K} to '{gpickle_path}'")

# Precompute adjacency powers
As = preprocess_adj_sparse(A_sparse, K)

