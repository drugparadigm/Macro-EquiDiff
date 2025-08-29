import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

# Input/output filenames
CSV_PATH = "macrocycle_table.csv"
SDF_PATH = "macrocycle_conformers.sdf"

# Parameters for conformer generation
NUM_CONFS = 1           # Only need one conformer per molecule for dataset prep
MAX_ATTEMPTS = 1000     # Increase if many macrocycles fail
RANDOM_SEED = 42

# Read CSV
df = pd.read_csv(CSV_PATH)
smiles_list = df['molecule'].unique()   # Use unique macrocycle SMILES

writer = Chem.SDWriter(SDF_PATH)

for smi in tqdm(smiles_list):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        print(f"Cannot parse SMILES: {smi}")
        continue

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = RANDOM_SEED
    success = AllChem.EmbedMolecule(mol, params)
    if success != 0:
        print(f"Conformer generation failed for: {smi}")
        continue

    AllChem.UFFOptimizeMolecule(mol)
    mol = Chem.RemoveHs(mol)
    mol.SetProp('_Name', smi)
    writer.write(mol)

writer.close()
print(f"Generated SDF: {SDF_PATH}")