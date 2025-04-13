import pickle
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

def main(network_file_path):
    # List to accumulate output lines
    output_lines = []
    
    # ----- Reloading the Network from Pickle -----
    try:
        with open(network_file_path, "rb") as f:
            G = pickle.load(f)
        output_lines.append("Network reloaded from " + network_file_path)
    except FileNotFoundError:
        output_lines.append("File not found: " + network_file_path)
        with open("network-stats.txt", "w") as outf:
            outf.write("\n".join(output_lines))
        return

    # ---------------------- 1. Basic Graph Statistics ----------------------
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    output_lines.append("Total Number of Nodes: " + str(n_nodes))
    output_lines.append("Total Number of Edges: " + str(n_edges))

    density = nx.density(G)
    output_lines.append("Graph Density: " + str(density))

    degrees = [d for _, d in G.degree()]
    avg_degree = np.mean(degrees)
    output_lines.append("Average Node Degree: " + str(avg_degree))

    # ---------------------- 2. Node Counts by Type ----------------------
    node_type_counts = {}
    for _, data in G.nodes(data=True):
        ntype = data.get("type", "Unknown")
        node_type_counts[ntype] = node_type_counts.get(ntype, 0) + 1

    output_lines.append("\nNode Type Counts:")
    for ntype, count in node_type_counts.items():
        output_lines.append("  " + str(ntype) + ": " + str(count))

    plt.figure(figsize=(8, 5))
    plt.bar(list(node_type_counts.keys()), list(node_type_counts.values()), color='skyblue')
    plt.xlabel("Node Type")
    plt.ylabel("Count")
    plt.title("Node Counts by Type")
    plt.tight_layout()
    plt.savefig("node_counts_by_type.png")
    plt.close()

    # ---------------------- 3. Edge Counts by Relation Type ----------------------
    edge_relation_counts = {}
    for _, _, data in G.edges(data=True):
        rel = data.get("relation", "Unknown")
        edge_relation_counts[rel] = edge_relation_counts.get(rel, 0) + 1

    output_lines.append("\nEdge Relation Counts:")
    for rel, count in edge_relation_counts.items():
        output_lines.append("  " + str(rel) + ": " + str(count))

    plt.figure(figsize=(8, 5))
    plt.bar(list(edge_relation_counts.keys()), list(edge_relation_counts.values()), color='salmon')
    plt.xlabel("Edge Relation Type")
    plt.ylabel("Count")
    plt.title("Edge Counts by Relation Type")
    plt.tight_layout()
    plt.savefig("edge_relation_counts.png")
    plt.close()

    # ---------------------- 4. Inferred Dataset Origin & Overlap Among Datasets ----------------------
    dataset_nodes = {"BindingDB": set(), "DTC": set(), "PharmGKB": set(), "BioPlex": set()}

    for u, v, data in G.edges(data=True):
        src = data.get("source", None)
        if src is None:
            rel = data.get("relation", "")
            if rel in ["drug-protein", "drug-variant", "protein-disease", "variant-disease"]:
                src = "PharmGKB"
            elif rel == "protein-protein":
                src = "BioPlex"
        if src not in dataset_nodes:
            dataset_nodes[src] = set()
        dataset_nodes[src].update([u, v])

    output_lines.append("\nDataset-specific Node Counts (via edge sources):")
    for ds, nodeset in dataset_nodes.items():
        output_lines.append("  " + str(ds) + ": " + str(len(nodeset)))

    dataset_names = list(dataset_nodes.keys())
    overlap_matrix = np.zeros((len(dataset_names), len(dataset_names)), dtype=int)

    for i, ds1 in enumerate(dataset_names):
        for j, ds2 in enumerate(dataset_names):
            overlap = dataset_nodes[ds1].intersection(dataset_nodes[ds2])
            overlap_matrix[i, j] = len(overlap)

    overlap_df = pd.DataFrame(overlap_matrix, index=dataset_names, columns=dataset_names)
    output_lines.append("\nPairwise Overlap (number of common nodes) between Datasets:")
    output_lines.append(overlap_df.to_string())

    plt.figure(figsize=(8, 6))
    plt.imshow(overlap_matrix, cmap="viridis")
    plt.colorbar(label="Number of Overlapping Nodes")
    plt.xticks(ticks=np.arange(len(dataset_names)), labels=dataset_names, rotation=45)
    plt.yticks(ticks=np.arange(len(dataset_names)), labels=dataset_names)
    plt.title("Overlap Between Dataset Node Sets")
    plt.tight_layout()
    plt.savefig("overlap_heatmap.png")
    plt.close()

    # Write all collected output lines to "network-stats.txt"
    with open("network-stats.txt", "w") as outf:
        outf.write("\n".join(output_lines))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph analysis script that loads a network from a pickle file and saves stats to a file.")
    parser.add_argument("network_file_path", help="Path to the network pickle file")
    args = parser.parse_args()
    main(args.network_file_path)
