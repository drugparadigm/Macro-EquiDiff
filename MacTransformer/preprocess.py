import pandas as pd
from rdkit import Chem,RDLogger
import random
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')

input_file = "datasets/input.txt"
output_file = "datasets/data.csv"

def remove_N_num(smiles):
    tokens = smiles.strip().split()
    if tokens and tokens[0].startswith("N_") and tokens[0][2:].isdigit():
        tokens = tokens[1:]
    return ''.join(tokens)

def is_valid_smiles(smiles):
    return Chem.MolFromSmiles(smiles) is not None

def generate_src_from_tgt(tgt):
    return tgt.replace("[*]", "").replace("(*)", "").replace("*", "")

def randomize_src(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    atom_order = list(range(mol.GetNumAtoms()))
    random.shuffle(atom_order)
    randomized_mol = Chem.RenumberAtoms(mol, atom_order)
    return Chem.MolToSmiles(randomized_mol, canonical=False, isomericSmiles=False)

def substructure_aligned_tsmiles(randomized_src, mol_smiles):
    randomized_src = randomized_src.replace('n1*', '[nH]1').replace('n2*', '[nH]2') \
                                   .replace('n3*', '[nH]3').replace('n(*)', '[nH]') \
                                   .replace('*n', '[nH]').replace('(*)', '').replace('*', '')

    frag = Chem.MolFromSmiles(randomized_src)
    mol = Chem.MolFromSmiles(mol_smiles)

    if frag is None or mol is None:
        return mol_smiles

    mol_atoms = list(range(mol.GetNumAtoms()))
    matches = mol.GetSubstructMatches(frag)

    if not matches:
        return mol_smiles

    for match in matches:
        match = list(match)
        link_atoms = [a for a in mol_atoms if a not in match]
        new_order = match + link_atoms
        new_mol = Chem.RenumberAtoms(mol, new_order)
        return Chem.MolToSmiles(new_mol, canonical=False, isomericSmiles=False)

    return mol_smiles

# Read input lines
with open(input_file, "r") as f:
    lines = f.read().splitlines()

rows = []
for line in tqdm(lines[:200000], desc="🧪 Processing SMILES"):
    # Step 1: Clean tgt
    canonical_tgt = remove_N_num(line)

    # Step 2: Randomize src (for alignment logic)
    randomized_src = randomize_src(generate_src_from_tgt(canonical_tgt))

    # Step 3: Augment tgt using alignment
    augmented_tgt = substructure_aligned_tsmiles(randomized_src, canonical_tgt)

    # Step 4: Generate src from augmented tgt
    src = generate_src_from_tgt(canonical_tgt)

    # Step 5: Validate
    if is_valid_smiles(src) and is_valid_smiles(augmented_tgt):
        rows.append({"src": src, "tgt": augmented_tgt})

# Save to CSV
df = pd.DataFrame(rows)
df.to_csv(output_file, index=False)
print(f"✅ Saved {len(df)} aligned and augmented SMILES pairs to {output_file}")
