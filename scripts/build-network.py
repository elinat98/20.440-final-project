import json
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
import os 
import pickle 


# ---------------------------------------------------
# 1. Load Mapping Files and Datasets
# ---------------------------------------------------
# Load the new mapping for PharmGKB to CHEMBL
with open("data/pharmgkb_to_chembl.json", "r") as f:
    pharmgkb_to_chembl = json.load(f)  # Maps PharmGKB drug IDs to CHEMBL IDs

# Load the remaining mapping files
with open("data/drug_pubchem_map.json", "r") as f:
    drug_pubchem_map = json.load(f)
with open("data/bindingdb_pubchem_to_chembl.json", "r") as f:
    bindingdb_pubchem_to_chembl = json.load(f)
with open("data/map_gene_uniprot.json", "r") as f:
    gene_uniprot_map = json.load(f)

# ---------------------------------------------------
# Datasets from PharmGKB (now TSV files with tab separator)
# ---------------------------------------------------
genes_df = pd.read_csv("data/genes.tsv", sep="\t")
drugs_df = pd.read_csv("data/drugs.tsv", sep="\t")
variants_df = pd.read_csv("data/variants.tsv", sep="\t")
relationships_df = pd.read_csv("data/relationships.tsv", sep="\t")

# ---------------------------------------------------
# Protein-protein interactions from BioPlex
# ---------------------------------------------------
bioplex_df = pd.read_csv("data/BioPlex.csv")

# ---------------------------------------------------
# BindingDB dataset (drug–target interactions using PubChem identifiers)
# ---------------------------------------------------
bindingdb_df = pd.read_csv("data/BindingDB_PubChem.csv")

# ---------------------------------------------------
# DTC IC50 data
# ---------------------------------------------------
dtc_df = pd.read_csv("data/DTC_IC50.csv")

# ---------------------------------------------------
# 2. Normalize the PharmGKB-to-CHEMBL Mapping
# ---------------------------------------------------
# Normalize all CHEMBL IDs to uppercase and strip any extra whitespace.
norm_pharmgkb_to_chembl = {k: v.strip().upper() for k, v in pharmgkb_to_chembl.items() if v}
pharm_chembl_set = set(norm_pharmgkb_to_chembl.values())

# ---------------------------------------------------
# 3. Helper Functions for Canonical Drug Node IDs
# ---------------------------------------------------
def get_canonical_drug_id_from_pharm(entity_id):
    """
    For PharmGKB drugs, use the pharmgkb_to_chembl mapping (normalized).
    If a CHEMBL id is found, return a canonical node id "drug_<CHEMBL>".
    Otherwise, create an id with a PharmGKB-specific prefix.
    """
    chembl_id = norm_pharmgkb_to_chembl.get(entity_id, None)
    if chembl_id is None:
        canonical = f"drug_pharm_{entity_id}"
    else:
        canonical = f"drug_{chembl_id}"
    return canonical

def get_canonical_drug_id_from_bindingdb(row):
    """
    For BindingDB rows, use the provided CHEMBL id (normalized) and then check if either the
    PubChem CID or SID exists in bindingdb_pubchem_to_chembl mapping; if so, use that mapping.
    """
    chembl = row.get("ChEMBL ID of Ligand", None)
    # Ensure chembl is converted to a string (if it is not NaN) before stripping
    if chembl is not None and pd.notna(chembl):
        chembl = str(chembl).strip().upper()
    else:
        chembl = None

    pubchem_cid = row.get("PubChem CID", None)
    pubchem_sid = row.get("PubChem SID", None)

    mapped = None
    if pubchem_cid and pd.notna(pubchem_cid):
        pubchem_cid = str(pubchem_cid).strip()
        if pubchem_cid in bindingdb_pubchem_to_chembl:
            mapped_val = bindingdb_pubchem_to_chembl[pubchem_cid]
            if mapped_val:  # Only update if mapped_val is truthy (not None or empty)
                mapped = str(mapped_val).strip().upper()
    if not mapped and pubchem_sid and pd.notna(pubchem_sid):
        pubchem_sid = str(pubchem_sid).strip()
        if pubchem_sid in bindingdb_pubchem_to_chembl:
            mapped_val = bindingdb_pubchem_to_chembl[pubchem_sid]
            if mapped_val:
                mapped = str(mapped_val).strip().upper()

    if mapped:
        chembl = mapped
    if chembl is None:
        chembl = "UNKNOWN"
    return f"drug_{chembl}"


def get_canonical_drug_id_from_dtc(row):
    """
    For DTC-IC50 rows, use the compound_id field (expected to be a CHEMBL id), normalize it,
    and cross-check with the PharmGKB mapping so that overlapping drugs are unified.
    """
    chembl = row.get("compound_id", None)
    if chembl is None:
        chembl = "UNKNOWN"
    chembl = chembl.strip().upper()
    if chembl in pharm_chembl_set:
        canonical = f"drug_{chembl}"
    else:
        canonical = f"drug_{chembl}"
    return canonical

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


# ---------------------------------------------------
# 4. Build the Heterogeneous Network
# ---------------------------------------------------
G = nx.MultiDiGraph()


# Optionally, pre-add every drug from the PharmGKB mapping to guarantee overlap.
for pharm_id, chembl in norm_pharmgkb_to_chembl.items():
    canonical = f"drug_{chembl}"
    if not G.has_node(canonical):
        G.add_node(canonical, type="Chemical", name=f"PharmGKB drug {pharm_id}")

# --- 4A. Add edges from PharmGKB relationships ---
for _, row in relationships_df.iterrows():
    node1, type1 = map_entity(row["Entity1_id"], row["Entity1_type"], row["Entity1_name"])
    node2, type2 = map_entity(row["Entity2_id"], row["Entity2_type"], row["Entity2_name"])

    if not G.has_node(node1):
        G.add_node(node1, type=type1, name=row["Entity1_name"])
    if not G.has_node(node2):
        G.add_node(node2, type=type2, name=row["Entity2_name"])

    # Determine relation type
    if ("Chemical" in [row["Entity1_type"], row["Entity2_type"]] and
        "Gene" in [row["Entity1_type"], row["Entity2_type"]]):
        rel = "drug-protein"
    elif ("Chemical" in [row["Entity1_type"], row["Entity2_type"]] and
          "Variant" in [row["Entity1_type"], row["Entity2_type"]]):
        rel = "drug-variant"
    elif ("Gene" in [row["Entity1_type"], row["Entity2_type"]] and
          "Disease" in [row["Entity1_type"], row["Entity2_type"]]):
        rel = "protein-disease"
    elif ("Variant" in [row["Entity1_type"], row["Entity2_type"]] and
          "Disease" in [row["Entity1_type"], row["Entity2_type"]]):
        rel = "variant-disease"
    elif row["Entity1_type"] == "Gene" and row["Entity2_type"] == "Gene":
        rel = "protein-protein"
    else:
        rel = "other"

    # Explicitly set source="PharmGKB" for these relationships.
    G.add_edge(node1, node2, relation=rel,
               evidence=row["Evidence"], association=row["Association"],
               source="PharmGKB")

# --- 4B. Add protein–protein interactions from BioPlex (pInt >= 0.7) ---
for _, row in bioplex_df.iterrows():
    confidence = row["pInt"]
    if confidence >= 0.7:
        node1 = f"protein_{row['UniprotA']}"
        node2 = f"protein_{row['UniprotB']}"
        if not G.has_node(node1):
            G.add_node(node1, type="Gene", name=row["SymbolA"])
        if not G.has_node(node2):
            G.add_node(node2, type="Gene", name=row["SymbolB"])
        # Set source="BioPlex" for these edges.
        G.add_edge(node1, node2, relation="protein-protein",
                   confidence=confidence, source="BioPlex")

# --- 4C. Add drug–protein interactions from BindingDB ---
for _, row in bindingdb_df.iterrows():
    node_drug = get_canonical_drug_id_from_bindingdb(row)
    protein_uniprot = row["UniProt (SwissProt) Primary ID of Target Chain"]
    node_protein = f"protein_{protein_uniprot}"

    if not G.has_node(node_drug):
        G.add_node(node_drug, type="Chemical", name=row["BindingDB Ligand Name"])
    if not G.has_node(node_protein):
        G.add_node(node_protein, type="Gene", name=row["Target Name"])

    # Process Ki value
    ki = row.get("Ki (nM)", None)
    try:
        ki_val = float(ki)
    except (TypeError, ValueError):
        ki_val = None
    if ki_val is not None and ki_val > 0:
        pActivity = -np.log10(ki_val * 1e-9)
    else:
        pActivity = None

    edge_attr = {"relation": "drug-protein", "source": "BindingDB"}
    if pActivity is not None:
        edge_attr["pActivity"] = pActivity
    G.add_edge(node_drug, node_protein, **edge_attr)

# --- 4D. Add drug–protein interactions from DTC IC50 data ---
for _, row in dtc_df.iterrows():
    if row["standard_type"] == "IC50" and pd.notna(row["standard_value"]):
        node_drug = get_canonical_drug_id_from_dtc(row)
        target_id = row["target_id"]
        node_protein = f"protein_{target_id}"

        try:
            value = float(row["standard_value"])
            if row["standard_units"].strip().upper() == "NM":
                pActivity = -np.log10(value * 1e-9)
            else:
                pActivity = -np.log10(value)
        except Exception:
            pActivity = None

        if not G.has_node(node_drug):
            name = row["compound_name"] if pd.notna(row["compound_name"]) else row["compound_id"]
            G.add_node(node_drug, type="Chemical", name=name)
        if not G.has_node(node_protein):
            name = row["target_pref_name"] if pd.notna(row["target_pref_name"]) else target_id
            G.add_node(node_protein, type="Gene", name=name)

        edge_attr = {"relation": "drug-protein", "source": "DTC"}
        if pActivity is not None:
            edge_attr["pActivity"] = pActivity
        G.add_edge(node_drug, node_protein, **edge_attr)

# ---------------------------------------------------
# 5. Save the Network to pickle for Later Reloading
# ---------------------------------------------------

# Suppose G is your NetworkX graph
# Set the path where you want to save the graph
output_pickle = "network.gpickle"

# Save the graph to a file using pickle
with open(output_pickle, "wb") as f:
    pickle.dump(G, f)

print(f"Graph saved to {output_pickle}.")

