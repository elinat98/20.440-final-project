# Network Analysis Setup 
This guide explains how to set up your environment and run the network analysis code for a given network (saved in a .gpickle format)

## 1. Create a new Conda environment
Create a new Conda environment named "network_analysis" with Python 3.9 (or your preferred version)

  ```sh
conda create --name network_analysis python=3.9 -y
  ```

## 2. Activate the new environment
  ```sh
conda activate network_analysis
  ```
## 3. Install the required packages using Conda
  ```sh
 conda install -y networkx numpy matplotlib pandas pickle
 ```


## 4. Run the network analysis code for the given network.gpickle file 
```sh
python graph-network.py network.gpickle 
```

## 5. Examine the output files 
The code once executed will put out a `network-stats.txt` file with information on the number of nodes, edges, graph density, and average node degree as well as node type counts, edge relation counts, and data-set sepecific node counts as well as pairwise overlap between datasets. The script will also produce figures displaying the data in network-stats.txt such as `edge_relation_counts.png`, a histogram of edge relations types, `node_counts_by_type.png`, a histogram of node types, and `overlap_heatmap.png`, a heatmap showing overlap between datasets. 