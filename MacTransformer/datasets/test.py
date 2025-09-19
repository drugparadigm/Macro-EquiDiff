# Read the SMILES from test.txt, clean them, and save to cleaned.txt
from rdkit import Chem
input_file = "/home/macrocycles/MacTransformer/datasets/zinc_test.txt"
output_file = "/home/macrocycles/MacTransformer/datasets/final_test_dataset_with_*.txt"

with open(input_file, "r") as f:
    lines = f.readlines()

cleaned_smiles = []
for line in lines:
    # Remove leading/trailing whitespace
    smile = line.strip()
    # Remove any N_int (like N_5, N_10, etc.)
    import re
    smile = re.sub(r'N_\d+', '', smile)
    # Remove spaces
    smile = smile.replace(' ', '')
    # Remove asterisks
    smile2 = smile.replace('[*]','').replace('(*)','').replace('*', '')
    if Chem.MolFromSmiles(smile2):
        cleaned_smiles.append(smile)

# Write cleaned SMILES to new file
with open(output_file, "w") as f:
    for sm in cleaned_smiles:
        f.write(sm + "\n")

print(f"Cleaned {len(cleaned_smiles)} SMILES saved to {output_file}")
