from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs,rdMolDescriptors
from collections import Counter
import pandas as pd

def load_smiles(file_path):
    """Load SMILES from a file, strip whitespace."""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def validate_smiles(smiles_list):
    """Return list of (original_smiles, mol) where mol is RDKit Mol or None."""
    return [(smi, Chem.MolFromSmiles(smi)) for smi in smiles_list]

def canonicalize_mol(mol):
    """Return canonical SMILES for a valid molecule."""
    return Chem.MolToSmiles(mol, canonical=True)

def calculate_tanimoto(mol1, mol2):
    """Calculate Tanimoto similarity between two mols."""
    fp1 = rdMolDescriptors.GetMACCSKeysFingerprint(mol1)
    fp2 = rdMolDescriptors.GetMACCSKeysFingerprint(mol2)
    return DataStructs.TanimotoSimilarity(fp1, fp2)

# === File paths ===
file1 = "datasets/predictions.txt"
file2 = "datasets/final_test_dataset.txt"

# Load SMILES
smiles1 = load_smiles(file1)
smiles2 = load_smiles(file2)
df=pd.read_csv("datasets/output.csv")

inp_train_data = df['tgt'].dropna().tolist()


# === Remove entries where file2 has duplicates (keep first occurrence) ===
seen = set()
filtered_smiles1 = []
filtered_smiles2 = []
for s1, s2 in zip(smiles1, smiles2):
    if s2 not in seen:
        seen.add(s2)
        filtered_smiles1.append(s1)
        filtered_smiles2.append(s2)

smiles1, smiles2 = filtered_smiles1, filtered_smiles2 

# Validate SMILES in file1
validated1 = validate_smiles(smiles1)
valid_mols1 = [(smi, mol) for smi, mol in validated1 if mol is not None]
num_valid = len(valid_mols1)
percent_valid = (num_valid / len(smiles1)) * 100 if smiles1 else 0

# Unique + duplicates
canonical_list = [canonicalize_mol(mol) for smi, mol in valid_mols1]
count_canon = Counter(canonical_list)
duplicates = {smi: count for smi, count in count_canon.items() if count > 1}
percent_unique = (len(count_canon) / num_valid) * 100 if num_valid else 0

# Tanimoto similarity (line-by-line, both valid)
tanimoto_values = []
for smi1, smi2 in zip(smiles1, smiles2):
    smi1.replace("(*)","").replace("[*]","").replace("*","")
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)
    if mol1 is not None and mol2 is not None:
        tanimoto_values.append(calculate_tanimoto(mol1, mol2))

avg_tanimoto = sum(tanimoto_values) / len(tanimoto_values) if tanimoto_values else 0
novel_list = [smi for smi in smiles1 if smi not in inp_train_data]
# === Results ===
print(f"Total in predictions: {len(smiles1)}")
print(f"Valid SMILES in predictions: {num_valid} ({percent_valid:.9f}%)")
print(f"Unique valid SMILES in predictions: {len(count_canon)} ({percent_unique:.9f}%)")
print(f"Average Tanimoto similarity: {avg_tanimoto:.9f}")
print(f"Novality: {len(novel_list)*100/len(smiles1)}%")

if duplicates:
    print("\nRepeated SMILES in file1 (canonical form):")
    for smi, count in duplicates.items():
        print(f"{smi} — {count} times")
else:
    print("\nNo repeated SMILES in file1.")
