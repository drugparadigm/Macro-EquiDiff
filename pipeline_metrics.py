"""import pandas as pd
from rdkit import Chem

with open("top_smiles5000.txt", "r") as f:
    smiles_list = [line.strip() for line in f if line.strip()]

total_generated = 5531
print(f"Total generated molecules: {total_generated}")

# 1️⃣ Check validity
valid_mols = []
for smi in smiles_list:
    mol = Chem.MolFromSmiles(smi)
    if mol:  # valid SMILES
        valid_mols.append(smi)

valid_count = len(valid_mols)
validity = (valid_count / total_generated) * 100 if total_generated > 0 else 0
print(f"Validity: {validity:.2f}%")

# 2️⃣ Check uniqueness among valid molecules
unique_valid_mols = list(set(valid_mols))
unique_count = len(unique_valid_mols)
uniqueness = (unique_count / valid_count) * 100 if valid_count > 0 else 0
print(f"Uniqueness: {uniqueness:.2f}%")

# 3️⃣ Check macrocyclization (macrocycle if ring size > 8 atoms)
macrocycles = []
for smi in unique_valid_mols:
    mol = Chem.MolFromSmiles(smi)
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    if any(len(ring) > 8 for ring in atom_rings):  # macrocycle definition
        macrocycles.append(smi)

macro_count = len(macrocycles)
macrocyclization = (macro_count / unique_count) * 100 if unique_count > 0 else 0
print(f"Macrocyclization: {macrocyclization:.2f}%")

"""
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, MolSurf, QED
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
import pandas as pd

# ---- Synthetic Accessibility Score (from Ertl & Schuffenhauer, 2009) ----
from rdkit.Chem import rdMolDescriptors
import math
from DiffLinker.src.delinker_utils.sascorer import calculateScore as calculate_sa_score

# ---- Property Functions ----
def calculate_physicochemical_properties(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Calculate properties
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        psa = MolSurf.TPSA(mol)
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        synthetic_accessibility = calculate_sa_score(mol)
        
        return {
            'SMILES': smiles,
            'MW': mw,
            'LogP': logp,
            'HBD': hbd,
            'HBA': hba,
            'PSA': psa,
            'RotatableBonds': rotatable_bonds,
            'SA': synthetic_accessibility
        }
    except:
        return None


def predict_admet_properties(mol):
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    psa = MolSurf.TPSA(mol)
    
    permeability = 'High' if logp > 2.5 and psa < 250 else 'Low'
    hba = Lipinski.NumHAcceptors(mol)
    hbd = Lipinski.NumHDonors(mol)
    herg_risk = 'High' if logp > 3 and hba > 3 else 'Low'
    cyp_risk = 'Low' if mw < 800 and hba < 10 else 'Moderate'
    num_heteroatoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in [6, 1])
    stability = 'High' if num_heteroatoms < 10 else 'Moderate'
    
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog(params)
    pains_alert = 'Yes' if catalog.HasMatch(mol) else 'No'
    lipinski = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    
    return {
        'Permeability': permeability,
        'hERG_Risk': herg_risk,
        'CYP_Inhibition': cyp_risk,
        'Metabolic_Stability': stability,
        'PAINS_Alert': pains_alert,
        'Lipinski_Violations': lipinski,
        'QED': QED.qed(mol)
    }


# ---- Run for both SMILES ----
smiles_list = [
    "c1cc2cc(c1O)CCC(=O)/C=C\\COCCC(=O)N2",
    "c1c2c(ccc1O/C=C\\CC(=O)CC/C=C\\CC2)NC(=O)C"
]

results = []
for smi in smiles_list:
    props = calculate_physicochemical_properties(smi)
    mol = Chem.MolFromSmiles(smi)
    admet = predict_admet_properties(mol)
    props.update(admet)
    results.append(props)

# ---- Export to CSV ----
df = pd.DataFrame(results)
df.to_csv("molecule_properties.csv", index=False)

print(df)

"""
import pandas as pd
from rdkit import Chem

# === CONFIG ===
csv_file = "molecules.csv"   # CSV file containing SMILES
smiles_column = "smiles"     # Name of the column with SMILES

# === LOAD DATA ===
df = pd.read_csv(csv_file)

# Store original count
total_generated = len(df)

# === VALIDITY ===
# Parse SMILES with RDKit to check validity
df["mol"] = df[smiles_column].apply(lambda s: Chem.MolFromSmiles(str(s)) if pd.notna(s) else None)
valid_df = df[df["mol"].notna()]
valid_count = len(valid_df)
validity = (valid_count / total_generated) * 100 if total_generated > 0 else 0

# === UNIQUENESS ===
unique_valid_smiles = set(valid_df[smiles_column])
unique_count = len(unique_valid_smiles)
uniqueness = (unique_count / valid_count) * 100 if valid_count > 0 else 0

# === MACROCYCLIZATION ===
def is_macrocycle(mol):
    if mol is None:
        return False
    ring_info = mol.GetRingInfo()
    # Check if any ring size is >= 12 atoms
    return any(len(ring) >= 12 for ring in ring_info.AtomRings())

macrocycle_count = sum(is_macrocycle(mol) for mol in valid_df.drop_duplicates(subset=[smiles_column])["mol"])
macrocyclization = (macrocycle_count / unique_count) * 100 if unique_count > 0 else 0

# === PRINT RESULTS ===
print(f"Validity: {validity:.2f}%")
print(f"Uniqueness: {uniqueness:.2f}%")
print(f"Macrocyclization: {macrocyclization:.2f}%")
"""

"""
# difflinker
import os
import numpy as np
from rdkit import Chem

inter_dir = "testing"  # base dir

skip_ids = {
    106, 126, 133, 138, 188, 218, 236, 252, 28, 398, 404, 416, 43, 458, 484, 488,
    530, 547, 557, 57, 577, 59, 630, 649, 657, 675, 688, 691, 705, 731, 74, 749,
    752, 759, 807, 830, 855, 874, 88, 916, 935, 963, 971, 984, 993, 995
}

# Per-input metric lists
validity_list = []
uniqueness_list = []
macrocycle_list = []

for input_id in range(0, 1000):
    if input_id in skip_ids:
        continue

    ops_dir = os.path.join(inter_dir, f"input_{input_id}", "diff_linker_ops")
    if not os.path.exists(ops_dir):
        continue

    sdf_files = sorted([f for f in os.listdir(ops_dir) if f.endswith(".sdf")])
    # sdf_files = sdf_files[:10]  # first 10 ops

    total_mols = 0
    valid_mols = []
    unique_valid_smiles = set()

    # Read molecules
    for sdf_file in sdf_files:
        sdf_path = os.path.join(ops_dir, sdf_file)
        suppl = Chem.SDMolSupplier(sdf_path, sanitize=False)
        for mol in suppl:
            total_mols += 1
            if mol is None:
                continue
            try:
                Chem.SanitizeMol(mol)
            except:
                continue
            smiles = Chem.MolToSmiles(mol)
            valid_mols.append(smiles)
            unique_valid_smiles.add(smiles)

    if total_mols == 0:
        continue

    # Macrocycle detection
    macrocycles_count = 0
    for smi in unique_valid_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            ring_info = mol.GetRingInfo()
            for ring in ring_info.AtomRings():
                if len(ring) >= 12:
                    macrocycles_count += 1
                    break

    # Per-input metrics
    validity = (len(valid_mols) / total_mols * 100) if total_mols else 0
    uniqueness = (len(unique_valid_smiles) / len(valid_mols) * 100) if valid_mols else 0
    macrocycle_percentage = (macrocycles_count / len(unique_valid_smiles) * 100) if unique_valid_smiles else 0

    validity_list.append(validity)
    uniqueness_list.append(uniqueness)
    macrocycle_list.append(macrocycle_percentage)

# Mean ± std for all inputs
def mean_std_str(values):
    return f"{np.mean(values):.2f}% ± {np.std(values, ddof=1):.2f}%"
print(validity_list)
print("Validity:", mean_std_str(validity_list))
print("Uniqueness:", mean_std_str(uniqueness_list))
print("Macrocyclization:", mean_std_str(macrocycle_list))

# import os
# import glob

# # Path to your main directory
# inter_dir = "testing"  # <-- change this

# # Output file
# output_file = os.path.join(inter_dir, "extracted_linkers_1000.txt")

# all_linkers = []

# # Loop through input_0 to input_999
# for folder in sorted(glob.glob(os.path.join(inter_dir, "input_*"))):
#     linkers_file = os.path.join(folder, "extracted_linkers.txt")
#     if os.path.isfile(linkers_file):
#         with open(linkers_file, "r") as f:
#             lines = [line.strip() for line in f if line.strip()]
#             all_linkers.extend(lines)

# # Save all collected linkers into one file
# with open(output_file, "w") as out:
#     for linker in all_linkers:
#         out.write(linker + "\n")

# print(f"Collected {len(all_linkers)} linkers into {output_file}")
"""
"""
#extract top1 linker -not stored in csv
import os
import itertools
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdchem

# ---------- CONFIG ----------
BASE_DIR = "testing"  # root containing input_0, input_1, ...
OUTPUT_FILE = "top1_linkers_1000.txt"
# Row index for "top-1": user said 2nd row; fall back to first if not enough rows.
TOP1_ROW_INDEX = 1

CYC_FILES = ["cyclizer_smiles.txt"]
LNK_FILES = ["extracted_linkers.txt"]
CSV_FILES = [ "top_5_macrors.csv"]
SMILES_COL_CANDIDATES = [ "SMILES"]

# ---------- UTILS ----------
def read_first_existing(path, candidates):
    for name in candidates:
        fp = os.path.join(path, name)
        if os.path.exists(fp):
            return fp
    return None

def get_smiles_column(df):
    cols_lower = {c.lower(): c for c in df.columns}
    for c in SMILES_COL_CANDIDATES:
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    raise ValueError(f"No SMILES column among: {df.columns.tolist()}")

def canon(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    Chem.SanitizeMol(m)
    return Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)

def canon_from_mol(mol):
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        # try with sanitizeOps=None then sanitize with default
        mol = Chem.Mol(mol)
        Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

def find_star_atoms(mol):
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]

def single_neighbor_idx(mol, star_idx):
    star = mol.GetAtomWithIdx(star_idx)
    nbs = list(star.GetNeighbors())
    if len(nbs) != 1:
        return None  # unexpected; skip if not exactly one neighbor
    return nbs[0].GetIdx()

def fuse_scaffold_and_linker(scaf_smiles, link_smiles):
    
    # Returns a list of RDKit molecules for both possible star-pair mappings.
    # Assumes exactly two '*' in scaffold and two '*' in linker.
    
    scaf = Chem.MolFromSmiles(scaf_smiles)
    link = Chem.MolFromSmiles(link_smiles)
    if scaf is None or link is None:
        return []

    s_stars = find_star_atoms(scaf)
    l_stars = find_star_atoms(link)
    if len(s_stars) != 2 or len(l_stars) != 2:
        return []

    # neighbor indices in their local mols
    s_nbs = [single_neighbor_idx(scaf, i) for i in s_stars]
    l_nbs = [single_neighbor_idx(link, i) for i in l_stars]
    if any(nb is None for nb in s_nbs + l_nbs):
        return []

    results = []
    for perm in [(0,1), (1,0)]:
        # combine molecules
        combo = Chem.CombineMols(link, scaf)
        rw = Chem.RWMol(combo)

        link_offset = 0
        scaf_offset = link.GetNumAtoms()

        # compute combined indices for link neighbors / stars
        l_star_idx_comb = [idx + link_offset for idx in l_stars]
        l_nb_idx_comb   = [idx + link_offset for idx in l_nbs]

        # combined indices for scaffold neighbors / stars
        s_star_idx_comb = [idx + scaf_offset for idx in s_stars]
        s_nb_idx_comb   = [idx + scaf_offset for idx in s_nbs]

        # Add bonds between corresponding neighbors (before star deletion)
        # linker star 0 -> scaffold star perm[0], linker star 1 -> scaffold star perm[1]
        rw.AddBond(l_nb_idx_comb[0], s_nb_idx_comb[perm[0]], rdchem.BondType.SINGLE)
        rw.AddBond(l_nb_idx_comb[1], s_nb_idx_comb[perm[1]], rdchem.BondType.SINGLE)

        # Remove stars (order by descending index to keep indices valid)
        to_remove = sorted(l_star_idx_comb + s_star_idx_comb, reverse=True)
        for idx in to_remove:
            rw.RemoveAtom(idx)

        try:
            mol = rw.GetMol()
            # Sanitize may fail if chemistry impossible; skip those
            Chem.SanitizeMol(mol)
            results.append(mol)
        except Exception:
            pass

    return results

# ---------- MAIN ----------
def main():
    out_lines = []
    for name in sorted(os.listdir(BASE_DIR)):
        inp_dir = os.path.join(BASE_DIR, name)
        if not os.path.isdir(inp_dir):
            continue
        if not name.startswith("input_"):
            continue

        cyc_fp = read_first_existing(inp_dir, CYC_FILES)
        lnk_fp = read_first_existing(inp_dir, LNK_FILES)
        csv_fp = read_first_existing(inp_dir, CSV_FILES)

        if not (cyc_fp and lnk_fp and csv_fp):
            out_lines.append(f"{name}\tMISSING_FILES")
            continue

        # read top-1 macro smiles
        try:
            df = pd.read_csv(csv_fp)
            smi_col = get_smiles_column(df)
            if len(df) == 0:
                out_lines.append(f"{name}\tCSV_EMPTY")
                continue
            row_idx = TOP1_ROW_INDEX if len(df) > TOP1_ROW_INDEX else 0
            top_macro = str(df.iloc[row_idx][smi_col]).strip()
            top_macro_can = canon(top_macro)
            if top_macro_can is None:
                out_lines.append(f"{name}\tTOP1_INVALID")
                continue
        except Exception as e:
            out_lines.append(f"{name}\tCSV_ERROR:{e}")
            continue

        # read scaffolds and linkers
        with open(cyc_fp, "r") as f:
            scaffolds = [ln.strip() for ln in f if ln.strip()]
        with open(lnk_fp, "r") as f:
            linkers = [ln.strip() for ln in f if ln.strip()]

        found = None

        for scaf in scaffolds:
            for lnk in linkers:
                # quick sanity: both must have exactly two '*'
                if scaf.count('*') != 2 or lnk.count('*') != 2:
                    continue
                for fused in fuse_scaffold_and_linker(scaf, lnk):
                    fused_can = canon_from_mol(fused)
                    if fused_can == top_macro_can:
                        found = lnk
                        break
                if found:
                    break
            if found:
                break

        if found is None:
            out_lines.append(f"{name}\tNO_MATCH")
        else:
            out_lines.append(f"{name}\t{found}")

    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(out_lines))

if __name__ == "__main__":
    main()
"""
"""
import os
import pandas as pd

def extract_linkers(base_dir="testing_new", output_file="l_check.txt", max_inputs=100):
    results = []

    for i in range(max_inputs+1):  # 0 to 60
        input_dir = os.path.join(base_dir, f"input_{i}")
        csv_path = os.path.join(input_dir, "top_5_macrors.csv")

        if not os.path.exists(csv_path):
            print(f"⚠ Skipping input_{i}: top_macros.csv not found")
            continue

        try:
            df = pd.read_csv(csv_path)

            # Ensure "linker" column exists and there is at least 2 rows
            if "Linkers" not in df.columns:
                print(f"⚠ Skipping input_{i}: 'linker' column not found")
                continue
            if len(df) < 2:
                print(f"⚠ Skipping input_{i}: less than 2 rows in CSV")
                continue

            # row 2 (second row → index 1)
            linker_smiles = df.loc[1, "Linkers"]
            results.append((f"input_{i}", linker_smiles))

        except Exception as e:
            print(f"❌ Error reading {csv_path}: {e}")

    # Write results to file
    with open(output_file, "w") as f:
        for input_id, linker in results:
            f.write(f"{input_id}\t{linker}\n")

    print(f"✅ Extracted {len(results)} linkers → {output_file}")


# Example usage
extract_linkers(base_dir="final_test_files/final_test", output_file="l_check5000.txt", max_inputs=6000)"""
