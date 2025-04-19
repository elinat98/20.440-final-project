import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
import json
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim 

import concurrent.futures
import requests


# -------------------------------
# 1. Data Loading and Mapping
# -------------------------------
# Correct file paths using /content/drive/MyDrive/...
with open("data/drug_chembl_map.json", "r") as f:
    drug_chembl_map = json.load(f)
with open("data/map_gene_uniprot.json", "r") as f:
    gene_uniprot_map = json.load(f)

# Load PharmGKB CSV/TSV files
genes_df = pd.read_csv("data/genes.tsv", sep="\t")
drugs_df = pd.read_csv("data/drugs.tsv", sep="\t")
variants_df = pd.read_csv("data/variants.tsv", sep="\t")
relationships_df = pd.read_csv("data/relationships.tsv", sep="\t")

# Load BioPlex protein–protein interactions
bioplex_df = pd.read_csv("data/BioPlex.csv")

# Load BindingDB data (drug–target interactions)
bindingdb_df = pd.read_csv("data/BindingDB_PubChem.csv")

# Load DTC IC50 data
dtc_df = pd.read_csv("data/DTC_IC50.csv")


def get_chembl_from_inchikey(inchi_key):
    """
    Given an InChIKey, retrieve the corresponding CHEMBL ID using the UniChem
    connectivity endpoint.

    The function sends a POST request with:
      - "type": "inchikey"
      - "compound": the provided InChIKey
      - "searchComponents": True

    It then searches the 'sources' array for the entry that corresponds to CHEMBL.
    Returns the CHEMBL ID (as a string) if found, otherwise None.
    """
    url = "https://www.ebi.ac.uk/unichem/api/v1/connectivity"
    payload = {
        "type": "inchikey",
        "compound": inchi_key,
        "searchComponents": True
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Error during UniChem connectivity request for InChIKey {inchi_key}: {e}")
        return None

    chembl_id = None
    # Check that 'sources' exists in the response and scan through mappings.
    if "sources" in data:
        for source in data["sources"]:
            # In the provided response, CHEMBL is identified by:
            #   - an "id" equal to 1, or
            #   - a longName/shortName containing 'chembl' (case-insensitive)
            if source.get("id") == 1 or \
               "chembl" in source.get("longName", "").lower() or \
               "chembl" in source.get("shortName", "").lower():
                chembl_id = source.get("compoundId")
                break
    else:
        print(f"No 'sources' field found in the response for InChIKey {inchi_key}")
    print(chembl_id)
    return chembl_id

def process_bindingdb_record(row):
    """
    Processes one record (row) from the BindingDB dataframe.

    Expects the row to have:
      - "Ligand InChI Key": the ligand’s InChIKey.
      - "Compound PubChem CID": the compound’s PubChem CID.

    Returns:
       (pubchem_cid, chembl_id): A tuple of the PubChem CID and the found CHEMBL ID.
    """
    inchi_key = row["Ligand InChI Key"]
    pubchem_cid = row["PubChem CID"]
    chembl_id = get_chembl_from_inchikey(inchi_key)
    if chembl_id:
        print(f"InChIKey {inchi_key} (PubChem CID {pubchem_cid}) maps to CHEMBL ID: {chembl_id}")
    else:
        print(f"No CHEMBL mapping found for InChIKey {inchi_key} (PubChem CID {pubchem_cid})")
    return pubchem_cid, chembl_id

# Prepare an empty dictionary to hold the mapping:
# Keys: Compound PubChem CID, Values: CHEMBL ID.
bindingdb_pubchem_to_chembl = {}

# Use ThreadPoolExecutor to parallelize the conversion.
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # Create a list of rows from the dataframe to process in parallel.
    rows = [row for index, row in bindingdb_df.iterrows()]
    results = list(executor.map(process_bindingdb_record, rows))

# Build the mapping dictionary (only include entries where a CHEMBL ID was found)
for pubchem_cid, chembl_id in results:
    bindingdb_pubchem_to_chembl[pubchem_cid] = chembl_id

# Save the dictionary to a JSON file.
output_filename = 'bindingdb_pubchem_to_chembl.json'
with open(output_filename, 'w') as f:
    json.dump(bindingdb_pubchem_to_chembl, f, indent=4)

print(f"Mapping dictionary saved to {output_filename} with {len(bindingdb_pubchem_to_chembl)} entries.")



genes_df["UniProtKB"] = None
for i,r in genes_df.iterrows():
  refs = (r['Cross-references'])
  uniprot_index = refs.find("UniProtKB:")
  if uniprot_index != -1:
    uniprot = refs[uniprot_index+10:]
    if uniprot.find(",") != -1:
      uniprot_list = uniprot.split(", UniProtKB:")
      genes_df.at[i,'UniProtKB'] = ", ".join(uniprot_list)
    else:
      genes_df.at[i,'UniProtKB'] = uniprot

# Create a mapping DataFrame from PharmGKB Accession Id to UniProtKB
mapping_gene_df = genes_df[['PharmGKB Accession Id', 'UniProtKB']].copy()

# Optionally, drop rows where either PharmGKB or UniProtKB is missing
mapping_gene_df.dropna(subset=['PharmGKB Accession Id', 'UniProtKB'], inplace=True)

# Optionally, if you need a dictionary mapping, you can convert it as follows:
pharmgkb_to_uniprot = mapping_gene_df.set_index('PharmGKB Accession Id')['UniProtKB'].to_dict()


with open('data/map_gene_uniprot.json', 'w') as f:
    json.dump(pharmgkb_to_uniprot, f, indent=4)

from chembl_webresource_client.new_client import new_client

def convert_to_chembl(external_identifier):
    """
    Converts an external identifier (e.g., 'DrugBank:DB00203') to a CHEMBL ID
    by searching the ChEMBL database using the search() method.
    """
    try:
        source, identifier = external_identifier.split(":", 1)
    except ValueError:
        print(f"Invalid format for external identifier: {external_identifier}")
        return None

    molecule = new_client.molecule
    try:
        results = molecule.search(identifier)
    except Exception as e:
        print(f"Error querying ChEMBL for identifier {identifier}: {e}")
        return None

    if results:
        return results[0].get("molecule_chembl_id")
    else:
        print(f"CHEMBL ID not found for: {external_identifier}")
        return None

def process_reference(task):
    """
    Process one task, which is a tuple of (pharmgkb_id, ext_id). Returns a tuple of (pharmgkb_id, chembl_id).
    """
    pharmgkb_id, ext_id = task
    try:
        chembl_id = convert_to_chembl(ext_id)
        print(f"{ext_id} -> CHEMBL ID: {chembl_id}")
    except Exception as e:
        print(f"Error processing {ext_id}: {e}")
        chembl_id = None
    return (pharmgkb_id, chembl_id)

# Build a list of tasks from your drugs DataFrame.
# Each task is a tuple: (PharmGKB Accession Id, one external reference)
# Replace NaN with None for the entire dataframe
drugs_df = drugs_df.where(pd.notnull(drugs_df), None)
tasks = []
for i, row in drugs_df.iterrows():
    pharmgkb_id = row['PharmGKB Accession Id']
    refs = row['Cross-references']
    if refs is not None:
        # Split by comma and strip extra whitespace
        for ext_id in refs.split(","):
            tasks.append((pharmgkb_id, ext_id.strip()))

drug_chembl_map = {}
# Assume ic50_data["compound_id"] is available. Convert it to a set for fast lookup.
ic50_compounds = set(dtc_df["compound_id"].values)

# Use a ThreadPoolExecutor to run the conversion concurrently.
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(process_reference, tasks))

# Build the mapping: assign the chembl_id for each pharmgkb_id if it meets our criteria.
for pharmgkb_id, chembl_id in results:
    if chembl_id is not None and chembl_id in ic50_compounds:
        drug_chembl_map[pharmgkb_id] = chembl_id

print(len(drug_chembl_map))

# Save the dictionary to a JSON file.
with open('data/drug_chembl_map.json', 'w') as f:
    json.dump(drug_chembl_map, f, indent=4)

def extract_pubchem_ids(refs):
    """
    Given a string of cross-references, extract the PubChem Compound ID (CID) and
    PubChem Substance ID (SID).

    The function assumes that:
    - The references are comma-separated.
    - "PubChem Compound:ID" and/or "PubChem Substance:ID" formats are used.
    """
    pubchem_cid = None
    pubchem_sid = None

    # If the cross-references field is empty or None, return nothing.
    if not refs:
        return pubchem_cid, pubchem_sid

    # Split the cross-references on commas and iterate over each entry
    for ref in refs.split(","):
        ref = ref.strip()
        # Check for PubChem Compound information
        if ref.lower().startswith("pubchem compound"):
            try:
                # Splitting into identifier parts
                pubchem_cid = ref.split(":", 1)[1].strip()
            except IndexError:
                print(f"Could not parse PubChem Compound from: {ref}")
        # Check for PubChem Substance information
        elif ref.lower().startswith("pubchem substance"):
            try:
                pubchem_sid = ref.split(":", 1)[1].strip()
            except IndexError:
                print(f"Could not parse PubChem Substance from: {ref}")

    return pubchem_cid, pubchem_sid

# Build a dictionary to map PharmGKB id to PubChem CID and SID
drug_pubchem_map = {}

for i, row in drugs_df.iterrows():
    pharmgkb_id = row['PharmGKB Accession Id']
    refs = row['Cross-references']
    pubchem_cid, pubchem_sid = extract_pubchem_ids(refs)

    # If either identifier exists, add to our mapping
    if pubchem_cid or pubchem_sid:
        drug_pubchem_map[pharmgkb_id] = {
            "pubchem_cid": pubchem_cid,
            "pubchem_sid": pubchem_sid
        }
        print(f"{pharmgkb_id}: PubChem CID -> {pubchem_cid}, PubChem SID -> {pubchem_sid}")

# Print how many mappings were found
print("Total mappings found:", len(drug_pubchem_map))

# Save the mapping dictionary to a JSON file
with open('data/drug_pubchem_map.json', 'w') as f:
    json.dump(drug_pubchem_map, f, indent=4)
