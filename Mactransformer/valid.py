import pandas as pd
from rdkit import Chem
from rdkit.Chem import MolToSmiles
from tqdm import tqdm


def clean_smiles(smiles):
    """Removes (*) markers from the target SMILES."""
    return smiles.replace('(*)','').replace('[*]', '').replace('*', '')

def is_valid_smiles(smiles):
    """Check if SMILES is valid."""
    return Chem.MolFromSmiles(smiles) is not None

def are_equal_mols(smiles1, smiles2):
    """Compare canonical forms of two SMILES strings."""
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return False
    return MolToSmiles(mol1, canonical=True) == MolToSmiles(mol2, canonical=True)

def validate_csv(csv_path):
    df = pd.read_csv(csv_path)
    correct, wrong = 0, 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Validating"):
        src = row['src']
        tgt = row['tgt']

        src_mol = Chem.MolFromSmiles(src)
        tgt_clean = clean_smiles(tgt)
        tgt_mol = Chem.MolFromSmiles(tgt_clean)

        if src_mol is None or tgt_mol is None:
            wrong += 1
        elif are_equal_mols(src, tgt_clean):
            correct += 1
        else:
            wrong += 1

    print(f"✅ Correct pairs: {correct}")
    print(f"❌ Wrong pairs  : {wrong}")
    print(f"📊 Total checked: {correct + wrong}")

if __name__ == "__main__":
    validate_csv("datasets/data.csv")  # <- replace with your actual CSV path
