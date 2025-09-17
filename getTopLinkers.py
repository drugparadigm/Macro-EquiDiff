import pandas as pd

# Read the CSV
df = pd.read_csv("final_test/top_performing_per_input.csv")

# Extract the 'smiles' column
smiles_list = df["SMILES"].dropna().tolist()

# Write to a text file, one SMILES per line
with open("top_smiles5000.txt", "w") as f:
    for s in smiles_list:
        f.write(s + "\n")

print("SMILES extracted and written to smiles.txt")
