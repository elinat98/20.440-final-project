import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import pickle
import networkx as nx
import scipy.sparse as sp
import os


def map_entity(entity_id, entity_type, entity_name):
    """
    Generic mapping for PharmGKB entities.
    For chemicals, use the PharmGKB canonical id from the new mapping.
    """
    if entity_type == "Chemical":
        return get_canonical_drug_id_from_pharm(entity_id), "Chemical"
    elif entity_type == "Gene":
        mapped = gene_uniprot_map.get(entity_id, entity_id)
        return f"protein_{mapped}", "Gene"
    elif entity_type == "Disease":
        return f"disease_{entity_id}", "Disease"
    elif entity_type == "Variant":
        return f"variant_{entity_id}", "Variant"
    else:
        return f"{entity_type}_{entity_id}", entity_type

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

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# 1) Build a mapping from each unique relation string to an integer class
all_rels = {data.get("relation", "other") for _, _, data in G.edges(data=True)}
relation2id = {rel: idx for idx, rel in enumerate(sorted(all_rels))}

# 2) Collect every edge (u→v) and its relation label
edge_list = []
edge_labels = []
for u, v, data in G.edges(data=True):
    rel = data.get("relation", "other")
    if u in node_idx and v in node_idx:
        edge_list.append((node_idx[u], node_idx[v]))
        edge_labels.append(relation2id[rel])

if not edge_list:
    raise ValueError("No edges found in G that map into node_idx!")

# 3) Convert to tensors
src_indices = torch.tensor([e[0] for e in edge_list], dtype=torch.long)
dst_indices = torch.tensor([e[1] for e in edge_list], dtype=torch.long)
labels      = torch.tensor(edge_labels,     dtype=torch.long)

# 4) Train/test split (stratified by relation label)
train_idx, test_idx = train_test_split(
    torch.arange(len(labels)).numpy(),
    test_size=0.2,
    stratify=labels.numpy(),
    random_state=42
)
train_idx = torch.from_numpy(train_idx)
test_idx  = torch.from_numpy(test_idx)

train_src   = src_indices[train_idx]
train_dst   = dst_indices[train_idx]
train_labels= labels[train_idx]

test_src    = src_indices[test_idx]
test_dst    = dst_indices[test_idx]
test_labels = labels[test_idx]

# 5) Wrap into DataLoaders
batch_size = 64

train_dataset = TensorDataset(train_src, train_dst, train_labels)
train_loader  = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

test_dataset = TensorDataset(test_src, test_dst, test_labels)
test_loader  = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

print(f"Train edges: {len(train_dataset)}, Test edges: {len(test_dataset)}")

import pickle

# -----------------------------------------
# 6) Save the train/test splits to gpickle
# -----------------------------------------

train_data = {
    "src_indices": train_src,
    "dst_indices": train_dst,
    "labels":      train_labels
}
test_data = {
    "src_indices": test_src,
    "dst_indices": test_dst,
    "labels":      test_labels
}

with open("train_data.gpickle", "wb") as f:
    pickle.dump(train_data, f)
print(f"Saved train split ({len(train_src)} edges) to 'train_data.gpickle'")

with open("/test_data.gpickle", "wb") as f:
    pickle.dump(test_data, f)
print(f"Saved test split  ({len(test_src)} edges) to 'test_data.gpickle'")

