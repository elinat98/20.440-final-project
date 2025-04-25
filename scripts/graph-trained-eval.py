import pickle
import torch
import numpy as np
import scipy.sparse as sp
import networkx as nx
import umap
from umap import UMAP
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sklearn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix
)
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
class SIGNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, K):
        super().__init__()
        self.K = K
        self.W = nn.ModuleList([nn.Linear(in_dim, hidden_dim) for _ in range(K+1)])
        self.act = nn.PReLU()
    def forward(self, X, As):
        out = 0
        for k in range(self.K+1):
            A_k = As[k]
            if A_k.is_sparse:
                with torch.amp.autocast(device_type="cuda", enabled=False):
                    tmp = torch.sparse.mm(A_k, X.float())
                tmp = tmp.to(X.dtype)
            else:
                tmp = A_k @ X
            out = out + self.W[k](tmp)
        return self.act(out)

class MultiClassDecoder(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, num_classes)
    def forward(self, emb_u, emb_v):
        x = emb_u * emb_v
        return self.classifier(x)

class GNNModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, K, num_classes):
        super().__init__()
        self.encoder = SIGNEncoder(in_dim, hidden_dim, K)
        self.decoder = MultiClassDecoder(hidden_dim, num_classes)
    def encode(self, X, As):
        return self.encoder(X, As)
    def decode(self, Z, u_idx, v_idx):
        return self.decoder(Z[u_idx], Z[v_idx])

def preprocess_adj_sparse(A, K, device=None):
    """
    A: scipy.sparse CSR matrix
    K: number of hops
    device: torch device (e.g. "cuda" or "cpu")
    Returns: list of K+1 torch sparse_coo_tensors on `device`
    """
    # Add self-loops
    I = sp.eye(A.shape[0], format='csr')
    A_tilde = A + I

    # Compute D^{-1/2}
    d = np.array(A_tilde.sum(axis=1)).flatten()
    d_inv_sqrt = 1.0 / np.sqrt(d)
    D_inv_sqrt = sp.diags(d_inv_sqrt)

    # Normalize
    A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt

    # Build powers
    As_list = []
    current = sp.eye(A.shape[0], format='csr')
    for k in range(K+1):
        if k > 0:
            current = current.dot(A_norm)
        coo = current.tocoo()
        indices = torch.from_numpy(
            np.vstack((coo.row, coo.col)).astype(np.int64)
        )
        values = torch.from_numpy(coo.data.astype(np.float32))
        shape  = coo.shape
        sparse_tensor = torch.sparse_coo_tensor(
            indices, values, shape,
            dtype=torch.float32,
            device=device
        )
        As_list.append(sparse_tensor.coalesce())
    return As_list

# --- Paths (adjust as needed) ---
graph_path    = "network.gpickle"
adj_path      = "training_files/A_sparse_and_K.gpickle"
test_split    = "training_files/test_data.gpickle"
model_ckpt    = "checkpoints/ckpt_epoch_20.pth"
metrics_path  = "training_files/training_metrics.pkl"

# --- Load Graph & Features ---
with open(graph_path, 'rb') as f:
    G = pickle.load(f)
node_list = list(G.nodes())
node_idx  = {n:i for i,n in enumerate(node_list)}

# Assume node_features and pad_features defined earlier
X = torch.tensor(
    np.vstack([pad_features(node_features[n]) for n in node_list]), 
    dtype=torch.float32
)

# --- Load Adjacency & Precompute As ---
A_sparse, K = pickle.load(open(adj_path,'rb'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
As = preprocess_adj_sparse(A_sparse, K, device)

# --- Load Test Split ---
with open(test_split, 'rb') as f:
    data = pickle.load(f)
test_src = data["src_indices"].long()
test_dst = data["dst_indices"].long()
test_labels = data["labels"].long().numpy()

# --- Load Model ---
# --- Load Model (fixed key) ---
ckpt = torch.load(model_ckpt, map_location='cpu')
state_dict = ckpt.get('model', ckpt)  # use 'model' key, or fall back to the whole dict
num_classes = state_dict['decoder.classifier.weight'].shape[0]

model = GNNModel(1280, 256, K, num_classes).to(device)
model.load_state_dict(state_dict)
model.eval() 


# --- Compute Embeddings ---
X_dev = X.to(device)
Z = model.encode(X_dev, As).detach().cpu().numpy()

# --- Batch Inference ---
src_dev = test_src.to(device)
dst_dev = test_dst.to(device)
with torch.no_grad():
    Z_dev = torch.tensor(Z, dtype=torch.float32, device=device)
    logits_all = model.decode(Z_dev, src_dev, dst_dev)
    probs_all  = torch.softmax(logits_all, dim=1).cpu().numpy()
preds_all = np.argmax(probs_all, axis=1)

# --- Metrics Prep ---
# Relation mapping
all_rels = sorted({d.get("relation","other") for _,_,d in G.edges(data=True)})
relation2id = {rel:i for i,rel in enumerate(all_rels)}

# ROC & PR curves per class + macro
roc_data = {}
pr_data  = {}
for i, rel in enumerate(all_rels):
    y_true = (test_labels == i).astype(int)
    y_score = probs_all[:, i]
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    roc_data[rel] = (fpr, tpr, roc_auc)
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    pr_data[rel] = (rec, prec, auc(rec, prec))

# Confusion matrix
cm = confusion_matrix(test_labels, preds_all)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

# UMAP projection
reducer = umap.UMAP(n_components=2, random_state=42)
Z2d = reducer.fit_transform(Z)
node_types = [G.nodes[n]['type'] for n in node_list]

# Precision@K for a chosen class (e.g., class 0)
K_vals = list(range(5, 101, 5))
precision_at_k = []
for k in K_vals:
    idx = np.argsort(-probs_all[:,0])[:k]
    precision_at_k.append((test_labels[idx]==0).mean())

# Calibration (class 0)
bins = np.linspace(0, 1, 11)
binids = np.digitize(probs_all[:,0], bins) - 1
bin_true = [test_labels[binids==i].mean() if np.any(binids==i) else np.nan 
            for i in range(len(bins))]
bin_pred = bins

# --- Plotting Panel ---
sns.set_theme(style='whitegrid', context='talk', palette='tab10')
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# 1) ROC Curves
ax = axes[0,0]
for rel, (fpr, tpr, auc_val) in roc_data.items():
    ax.plot(fpr, tpr, label=f"{rel} (AUC={auc_val:.2f})")
ax.set(title="ROC Curves", xlabel="FPR", ylabel="TPR")
ax.legend(loc='lower right', fontsize='small')

# 2) Precision-Recall
ax = axes[0,1]
for rel, (rec, prec, ap) in pr_data.items():
    ax.plot(rec, prec, label=f"{rel} (AP={ap:.2f})")
ax.set(title="Precision-Recall", xlabel="Recall", ylabel="Precision")
ax.legend(loc='lower left', fontsize='small')

# 3) Confusion Matrix
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", ax=axes[0,2],
            xticklabels=all_rels, yticklabels=all_rels)
axes[0,2].set(title="Confusion Matrix", xlabel="Predicted", ylabel="True")

# 4) UMAP embedding
sns.scatterplot(x=Z2d[:,0], y=Z2d[:,1], hue=node_types, s=20, ax=axes[1,0],
                legend=False)
axes[1,0].set(title="UMAP of Node Embeddings")

# 5) Precision@K
sns.lineplot(x=K_vals, y=precision_at_k, marker='o', ax=axes[1,1])
axes[1,1].set(title="Precision@K (class 0)", xlabel="K", ylabel="Precision")

# 6) Calibration Curve
axes[1,2].plot(bin_pred, bin_true, marker='o', linewidth=2)
axes[1,2].plot([0,1], [0,1], '--', color='gray')
axes[1,2].set(title="Calibration Curve (class 0)", xlabel="Predicted", ylabel="True")

plt.tight_layout()
fig.savefig('evaluation_panel.png', dpi=300, bbox_inches='tight')
plt.show()