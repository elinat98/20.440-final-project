import os
import pickle
import numpy as np
import scipy.sparse as sp
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import pandas as pd
import pickle



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





# ----- Set Up Device -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----- 1) Reload Heterogeneous Graph -----
network_path = "network.gpickle"
if not os.path.exists(network_path):
    raise FileNotFoundError(f"{network_path} not found!")
with open(network_path, "rb") as f:
    G = pickle.load(f)
print("✔ Loaded graph")

# ----- 2) Rebuild Node List & Features (X) -----
node_list = list(G.nodes())
node_idx  = {n: i for i, n in enumerate(node_list)}

def pad_features(vec, target_dim=1280):
    if vec.shape[0] < target_dim:
        return np.pad(vec, (0, target_dim - vec.shape[0]), mode='constant')
    else:
        return vec[:target_dim]

# assume node_features dict already exists
X_list = [pad_features(node_features[n], target_dim=1280) for n in node_list]
X = torch.tensor(np.vstack(X_list), dtype=torch.float32).to(device)

# ----- 3) Reload Precomputed Adjacency & Hops -----
adj_path = "A_sparse_and_K.gpickle"
if not os.path.exists(adj_path):
    raise FileNotFoundError(f"{adj_path} not found!")
with open(adj_path, "rb") as f:
    A_sparse, K = pickle.load(f)
print(f"✔ Loaded adjacency (K={K})")
def preprocess_adj_sparse(A, K, device):
    I = sp.eye(A.shape[0], format="csr")
    A_tilde = A + I
    d = np.array(A_tilde.sum(axis=1)).flatten()
    d_inv_sqrt = 1.0 / np.sqrt(d)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt

    As_list = []
    current = sp.eye(A.shape[0], format="csr")
    for k in range(K+1):
        if k > 0:
            current = current.dot(A_norm)
        coo = current.tocoo()
        idx = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64))
        vals = torch.from_numpy(coo.data.astype(np.float32))
        tensor = torch.sparse_coo_tensor(
            idx, vals, coo.shape, dtype=torch.float32, device=device
        ).coalesce()
        As_list.append(tensor)
    return As_list

As = preprocess_adj_sparse(A_sparse, K, device)

# ----- 4) Build Relation Mapping & Reload Splits -----
all_rels = {data.get("relation", "other") for _, _, data in G.edges(data=True)}
relation2id = {rel: i for i, rel in enumerate(sorted(all_rels))}
num_classes = len(relation2id)
print(f"✔ {num_classes} relation classes")

def load_split(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found!")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return (
        data["src_indices"].cpu(),
        data["dst_indices"].cpu(),
        data["labels"].cpu().long()
    )

train_src, train_dst, train_labels = load_split("train_data.gpickle")
test_src,  test_dst,  test_labels  = load_split("test_data.gpickle")
print(f"✔ Loaded train ({len(train_src)}) / test ({len(test_src)}) edges")



# ----- 5) Create DataLoaders -----
batch_size = 64
train_loader = DataLoader(
    TensorDataset(train_src, train_dst, train_labels),
    batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True
)
test_loader = DataLoader(
    TensorDataset(test_src, test_dst, test_labels),
    batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
)

# ----- 6) Define the GNN + Decoder -----
in_dim, hidden_dim = 1280, 256

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

model = GNNModel(in_dim, hidden_dim, K, num_classes).to(device)

# ----- 7) Training Setup -----
optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)
criterion = nn.CrossEntropyLoss()
scaler    = torch.cuda.amp.GradScaler()
num_epochs = 50

# ----- 8) Train / Evaluate Loop -----
train_losses, train_accs = [], []
test_losses,  test_accs  = [], []

for epoch in range(1, num_epochs+1):
    # Precompute embeddings once per epoch
    model.train()
    Z = model.encode(X, As).detach()

    # Training
    rloss, correct, total = 0.0, 0, 0
    for u_idx, v_idx, lbl in tqdm(train_loader, desc=f"Train Epoch {epoch}", leave=False):
        u_idx, v_idx, lbl = (
            u_idx.to(device, non_blocking=True),
            v_idx.to(device, non_blocking=True),
            lbl.to(device, non_blocking=True),
        )
        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda"):
            logits = model.decode(Z, u_idx, v_idx)
            loss = criterion(logits, lbl)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        rloss += loss.item() * lbl.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == lbl).sum().item()
        total += lbl.size(0)
    train_losses.append(rloss/total)
    train_accs.append(correct/total)

    # Evaluation
    model.eval()
    rloss, correct, total = 0.0, 0, 0
    with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
        for u_idx, v_idx, lbl in test_loader:
            u_idx, v_idx, lbl = (
                u_idx.to(device, non_blocking=True),
                v_idx.to(device, non_blocking=True),
                lbl.to(device, non_blocking=True),
            )
            logits = model.decode(Z, u_idx, v_idx)
            loss = criterion(logits, lbl)
            rloss += loss.item() * lbl.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == lbl).sum().item()
            total += lbl.size(0)
    test_losses.append(rloss/total)
    test_accs.append(correct/total)

    print(
        f"Epoch {epoch}/{num_epochs} | "
        f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accs[-1]:.4f} | "
        f"Test Loss:  {test_losses[-1]:.4f}, Test Acc:  {test_accs[-1]:.4f}"
    )

    # Save checkpoint
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'opt': optimizer.state_dict(),
        'train_loss': train_losses[-1],
        'test_loss': test_losses[-1]
    }, f"{ckpt_dir}/ckpt_epoch_{epoch}.pth")

# ----- 9) Save Metrics -----
metrics = {
    'train_losses': train_losses,
    'train_accs':   train_accs,
    'test_losses':  test_losses,
    'test_accs':    test_accs
}
metrics_path = "training_metrics.pkl"
with open(metrics_path, 'wb') as f:
    pickle.dump(metrics, f)
print(f"Training complete. Metrics saved to '{metrics_path}'.")
