import pandas as pd
import re
from rdkit import Chem,  RDLogger
from rdkit.Chem import GetShortestPath
from itertools import combinations
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*') 

def singlebonds_in_macring(mol):
    singlebonds = []
    rings = list(mol.GetRingInfo().AtomRings())
    if not rings:
        return singlebonds
    rings.sort(key=lambda x: len(x))
    maxring = rings[-1]
    for b in mol.GetBonds():
        if b.GetBondType() == Chem.rdchem.BondType.SINGLE:
            id1 = b.GetBeginAtomIdx()
            id2 = b.GetEndAtomIdx()
            if id1 in maxring and id2 in maxring:
                singlebonds.append(b.GetIdx())
    return singlebonds


def filter_chain_length(chain):
    allatoms = chain.GetNumHeavyAtoms()
    dummy_ids = [a.GetIdx() for a in chain.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummy_ids) != 2:
        return False, 0
    length = len(GetShortestPath(chain, dummy_ids[0], dummy_ids[1])) - 2
    action = (allatoms > 2) and (2 < length < 10) and (length / allatoms >= 0.6)
    return action, length


def filter_chain_ring(chain):
    rings = chain.GetRingInfo().AtomRings()
    if len(rings) == 0:
        return True, 0
    elif len(rings) == 1:
        ring_numatom = max([len(k) for k in rings])
        return ring_numatom < 7, ring_numatom
    return False, 8


def remove_dummy_index(frag, pattern=re.compile(r'\[\d+.\]')):
    smiles = Chem.MolToSmiles(frag)
    dummy_index = pattern.findall(smiles)
    for i in dummy_index:
        smiles = smiles.replace(i, '[*]')
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def fragmentation_details(mol, original_smiles):
    bonds = singlebonds_in_macring(mol)
    if len(bonds) == 0:
        return []

    records = []

    for b in combinations(bonds, 2):
        frags_mol = Chem.FragmentOnBonds(mol, list(b), addDummies=True)
        frags_mols = Chem.GetMolFrags(frags_mol, asMols=True)
        if len(frags_mols) != 2:
            continue

        frag1, frag2 = frags_mols
        if frag1.GetNumHeavyAtoms() < frag2.GetNumHeavyAtoms():
            chain, frag = frag1, frag2
        else:
            chain, frag = frag2, frag1

        if (chain.GetNumHeavyAtoms() / mol.GetNumHeavyAtoms()) < 0.25:
            pass_len, length = filter_chain_length(chain)
            pass_ring, ring_size = filter_chain_ring(chain)
            if pass_len and pass_ring:
                acyclic_with_dummy = remove_dummy_index(frag)
                linker = remove_dummy_index(chain)


                # Get anchor positions (dummy atoms)
                mol_frag = Chem.MolFromSmiles(acyclic_with_dummy)
                anchor_indices = [atom.GetIdx() for atom in mol_frag.GetAtoms() if atom.GetAtomicNum() == 0]
                if len(anchor_indices) == 2:
                    records.append({
                        "molecule": original_smiles,
                        "fragments": acyclic_with_dummy,
                        "linker": linker
                    })

    return records

def is_valid_smiles(smiles):
    """Checks if a SMILES string is valid."""
    return Chem.MolFromSmiles(smiles) is not None


def process_macrocycles_csv(input_csv_path, output_csv_path):
    df = pd.read_csv(input_csv_path, encoding='latin-1') # Added encoding='latin-1'
    all_records = []
    discarded_count = 0

    # Add tqdm to the loop
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing macrocycles"):
        smiles = row['SMILES at pH 7']
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            discarded_count += 1
            continue
        Chem.RemoveStereochemistry(mol)
        smiles = Chem.MolToSmiles(mol)


        records = fragmentation_details(mol, smiles)


        # Validate generated SMILES and filter records
        valid_records = []
        for record in records:
            # Also check if clean_acyclic is not an empty string after removing '*'
            if (is_valid_smiles(record["fragments"]) and
                is_valid_smiles(record["linker"])):
                valid_records.append(record)
            else:
                discarded_count += 1 # Count records with invalid generated SMILES or empty clean_acyclic

        all_records.extend(valid_records)

    # Create DataFrame and remove duplicates
    out_df = pd.DataFrame(all_records)
    out_df.drop_duplicates(inplace=True)

    out_df.to_csv(output_csv_path, index=False)
    print(f"Saved results to {output_csv_path}")
    print(f"Total rows discarded due to invalid initial or generated SMILES or empty clean_acyclic: {discarded_count}")



process_macrocycles_csv("data/jm3c00134_si_002.csv", "data/macrocycle_data.csv")

# Uncomment and modify the path below for direct execution
# if __name__ == "__main__":
#     process_macrocycles_csv("macrocycles.csv", "fragmented_macrocycles.csv")