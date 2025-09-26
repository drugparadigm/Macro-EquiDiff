import os
import subprocess
import tempfile
import argparse
import sys
from rdkit import Chem
from rdkit.Chem import AllChem, SDWriter
import numpy as np
from rdkit.Chem import SanitizeFlags
from rdkit.Chem import Descriptors, MolSurf, Crippen, Lipinski, QED
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
import pandas as pd
import torch


INTER_DIR = "intermediates"
N_AUGMENT = 3

try:
    from EDM.src.delinker_utils.sascorer import calculateScore as calculate_sa_score
except ImportError:
    def calculate_sa_score(_):
        print("NOOOOOOOOOOOOOOOOOOOOOOOOOOOoo")
        return None  
        
# ========== UTILS ==========
def smi_to_sdf(smi, out_sdf):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError("Invalid SMILES")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    writer = SDWriter(out_sdf)
    writer.write(mol)
    writer.close()

def neutralize_radicals(mol):
    """
    Neutralize radicals by converting unpaired electrons to hydrogen atoms.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        mol: Modified molecule with radicals neutralized
    """
    for atom in mol.GetAtoms():
        r = atom.GetNumRadicalElectrons()
        if r > 0:
            atom.SetNumRadicalElectrons(0)
            atom.SetNumExplicitHs(atom.GetNumExplicitHs() + r)
            atom.SetNoImplicit(False)
    return mol

def remove_radicals(smiles_input):

    # Process SMILES with radicals and return neutralized SMILES.
    
    # Args:
    #     smiles_input: Input SMILES string (may contain radicals)
        
    # Returns:
    #     str: Output SMILES string with radicals neutralized, or None if error

    try:
        # Parse SMILES to molecule (don't sanitize initially)
        mol = Chem.MolFromSmiles(smiles_input, sanitize=False)
        if mol is None:
            print(f"Invalid SMILES: {smiles_input}")
            return None
        
        # Assign radicals
        Chem.AssignRadicals(mol)
        
        # Check for radicals and report them
        radical_atoms = []
        for atom in mol.GetAtoms():
            r = atom.GetNumRadicalElectrons()
            if r > 0:
                radical_atoms.append((atom.GetIdx() + 1, r))
                print(f"Atom {atom.GetIdx()+1} ({atom.GetSymbol()}): {r} unpaired electron(s)")
        
        if not radical_atoms:
            print("No radicals found in the molecule")
            return smiles_input
        
        # Neutralize radicals
        mol = neutralize_radicals(mol)
        
        # Sanitize the molecule
        Chem.SanitizeMol(mol)
        
        # Generate output SMILES (without explicit hydrogens)
        output_smiles = Chem.MolToSmiles(mol)
        
        return output_smiles
        
    except Exception as e:
        print(f"Error processing SMILES '{smiles_input}': {e}")
        return None
    

def generate_augmented_smiles(smiles, n=2, unique=True):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    augmented = set()

    for _ in range(n):  # Try more than n to ensure uniquenes- n*5
        new_smiles =Chem.MolToSmiles(mol, doRandom=True)
        if unique:
            augmented.add(new_smiles)
            if len(augmented) >= n:
                break
        else:
            augmented.append(new_smiles)

    return list(augmented)[:n]

def run_macformer_on_smiles_in_memory(smiles, inter_dir):
    out_path = f"{inter_dir}/valid_macformer_smiles.txt"
    vocab_path = "MacTransformer/vocab.pt"

    subprocess.run([
        "conda", "run", "-n", "myenv", "--no-capture-output",
        "python", "MacTransformer/pipeline_predict.py",
        "--checkpoint", "MacTransformer/macformer_checkpoint_epoch_18.pth",
        "--smiles", smiles,  # NEW ARG: pass directly
        "--output_file", out_path,
        "--vocab", vocab_path,
        "--beam_width", "1"
    ], check=True)

    with open(out_path) as f:
        return [line.strip() for line in f if line.strip()]

def filter_valid_smiles(smiles_list):
    result = []
    seen = set()   # track unique SMILES

    for smile in smiles_list:
        # clean special tokens
        smile = smile.replace("<s>", "").replace("</s>", "")

        print(smile)

        if smile.count("*") == 2:
            mol = Chem.MolFromSmiles(smile)
            if mol:
                if smile not in seen:   # only add if not seen
                    result.append(smile)
                    seen.add(smile)
            else:
                print(f"⚠ Invalid SMILES: {smile}")
    return result


def run_EDM(inter_dir):
    subprocess.run([
        "conda", "run", "-n", "dl2", "--no-capture-output",
        "python", "-W", "ignore", "EDM/generate.py",
        "--fragments", f"{inter_dir}/user_input.sdf",
        "--model", "EDM/models/geom_EDM.ckpt",
        "--linker_size", "EDM/models/geom_size_gnn.ckpt",
        "--output", f"{inter_dir}/diff_linker_ops"
    ], check=True)

def load_mol(sdf_path):
    try:
        suppl = Chem.SDMolSupplier(sdf_path, sanitize=True, removeHs=False)
        for mol in suppl:
            if mol is not None:
                return mol
    except Exception as e:
        print(f"⚠  RDKit load error for {sdf_path}: {e}", file=sys.stderr)
    print(f"⚠  Could not load molecule from {sdf_path} (skipped)", file=sys.stderr)
    return None

def get_heavy_atom_info(mol):
    heavy_atoms = []
    conf = mol.GetConformer()
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        pos = conf.GetAtomPosition(atom.GetIdx())
        coord = np.array([pos.x, pos.y, pos.z])
        heavy_atoms.append((atom.GetSymbol(), coord, atom.GetIdx()))
    return heavy_atoms

def find_linker_atom_indices(acyclic_mol, macro_mol):
    # Find linker atoms by removing the number of heavy atoms in acyclic_mol from macro_mol.
    acyclic_num_heavy_atoms = sum(1 for atom in acyclic_mol.GetAtoms() if atom.GetAtomicNum() != 1)
    macro_num_atoms = macro_mol.GetNumAtoms()
    
    print(f"📄 Acyclic heavy atoms: {acyclic_num_heavy_atoms}, Macro atoms: {macro_num_atoms}, Potential linker atoms: {macro_num_atoms - acyclic_num_heavy_atoms}", file=sys.stderr)
    
    if acyclic_num_heavy_atoms >= macro_num_atoms:
        print(f"❌ Acyclic molecule has {acyclic_num_heavy_atoms} heavy atoms, which is >= macrocycle's {macro_num_atoms} atoms.", file=sys.stderr)
        return []

    # Assume the first acyclic_num_heavy_atoms in macro_mol correspond to the acyclic fragment
    linker_atom_indices = list(range(acyclic_num_heavy_atoms, macro_num_atoms))
    return linker_atom_indices


def find_anchor_points(macro_mol, linker_atom_indices):
    # Find linker atoms that were connected to non-linker (acyclic) atoms.
    linker_set = set(linker_atom_indices)
    anchor_points = set()
    for bond in macro_mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        if a1 in linker_set and a2 not in linker_set:
            anchor_points.add(a1)
        elif a2 in linker_set and a1 not in linker_set:
            anchor_points.add(a2)
    return list(anchor_points)

def select_farthest_anchor_points(macro_mol, anchor_points):
    # Select the two anchor points that are farthest apart, preferring non-ring atoms.
    if len(anchor_points) <= 2:
        return anchor_points

    # Get ring information
    ring_info = macro_mol.GetRingInfo()
    # Separate anchor points into ring and non-ring atoms
    non_ring_anchors = [idx for idx in anchor_points if not ring_info.IsAtomInRingOfSize(idx, ring_info.NumRings())]
    ring_anchors = [idx for idx in anchor_points if ring_info.IsAtomInRingOfSize(idx, ring_info.NumRings())]

    conf = macro_mol.GetConformer()
    max_dist = -1
    farthest_pair = None

    # Prefer non-ring anchors if available
    if len(non_ring_anchors) >= 2:
        # Calculate pairwise distances among non-ring anchors
        for i, idx1 in enumerate(non_ring_anchors):
            pos1 = conf.GetAtomPosition(idx1)
            coord1 = np.array([pos1.x, pos1.y, pos1.z])
            for idx2 in non_ring_anchors[i + 1:]:
                pos2 = conf.GetAtomPosition(idx2)
                coord2 = np.array([pos2.x, pos2.y, pos2.z])
                dist = np.linalg.norm(coord1 - coord2)
                if dist > max_dist:
                    max_dist = dist
                    farthest_pair = [idx1, idx2]
    else:
        # If fewer than 2 non-ring anchors, use all anchors and issue warning
        print(f"⚠  Fewer than 2 non-ring anchor points ({len(non_ring_anchors)}), using ring atoms if necessary", file=sys.stderr)
        for i, idx1 in enumerate(anchor_points):
            pos1 = conf.GetAtomPosition(idx1)
            coord1 = np.array([pos1.x, pos1.y, pos1.z])
            for idx2 in anchor_points[i + 1:]:
                pos2 = conf.GetAtomPosition(idx2)
                coord2 = np.array([pos2.x, pos2.y, pos2.z])
                dist = np.linalg.norm(coord1 - coord2)
                if dist > max_dist:
                    max_dist = dist
                    farthest_pair = [idx1, idx2]

    if farthest_pair is None:
        print(f"⚠  Could not find valid anchor point pair", file=sys.stderr)
        return None
    return farthest_pair
    
def add_second_attachment_point_and_return_indices(linker_mol):
    linker = Chem.RWMol(linker_mol)
    dummy_idxs = [atom.GetIdx() for atom in linker.GetAtoms() if atom.GetAtomicNum() == 0]

    # Case: Already has two or more attachment points
    if len(dummy_idxs) >= 2:
        return dummy_idxs[:2]  # Return only the first two

    # Helper to check if atom has free valence
    def atom_has_free_valence(atom):
        valence = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
        max_valence = Chem.GetPeriodicTable().GetDefaultValence(atom.GetAtomicNum())
        return valence < max_valence

    # Try to find an atom with free valence (non-dummy)
    attachable_atoms = [atom for atom in linker.GetAtoms()
                        if atom.GetAtomicNum() > 0 and atom_has_free_valence(atom)]

    if not attachable_atoms:
        print("❌ No atoms with available valence to attach new attachment point.")
        return None

    # Add second dummy atom
    target_atom = attachable_atoms[0]
    new_dummy_idx = linker.AddAtom(Chem.Atom(0))
    linker.AddBond(target_atom.GetIdx(), new_dummy_idx, Chem.BondType.SINGLE)

    # Return updated dummy atom indices (original + new)
    dummy_idxs = [atom.GetIdx() for atom in linker.GetMol().GetAtoms() if atom.GetAtomicNum() == 0]
    return dummy_idxs[:2]

def extract_linker_from_macro(macro_mol, linker_atom_indices):
    anchor_points = find_anchor_points(macro_mol, linker_atom_indices)
    print("🧠",anchor_points)
    if anchor_points is None or len(anchor_points) != 2:
        print(f"⭕  Expected exactly 2 anchor points, found {len(anchor_points)} after selection", file=sys.stderr)

        # Try to fix: Add second dummy if only one present
        if anchor_points is not None and len(anchor_points) == 1:
            print("🛠  Attempting to add second attachment point...")
            new_anchor_points = add_second_attachment_point_and_return_indices(macro_mol)
            anchor_points.append(new_anchor_points[0])
            print("##########",anchor_points)
            if new_anchor_points is None or len(new_anchor_points) != 2:
                print(f"❌  Failed to recover a second anchor point automatically", file=sys.stderr)
                return None
            else:
                anchor_points.append(new_anchor_points[0])
                
        else:
            return None
    selected_anchors = select_farthest_anchor_points(macro_mol, anchor_points)
    if selected_anchors is None or len(selected_anchors) != 2:
        print(f"⚠  Expected exactly 2 anchor points, found {len(selected_anchors)} after selection", file=sys.stderr)
        return None

    emol = Chem.EditableMol(Chem.Mol())
    old_to_new = {}
    
    for old_idx in linker_atom_indices:
        atom = macro_mol.GetAtomWithIdx(old_idx)
        new_idx = emol.AddAtom(Chem.Atom(atom.GetSymbol()))
        old_to_new[old_idx] = new_idx

    dummy_indices = []
    conf = macro_mol.GetConformer()
    for anchor_idx in selected_anchors:
        dummy_idx = emol.AddAtom(Chem.Atom(0))  # Atomic number 0 = *
        dummy_indices.append(dummy_idx)
        emol.AddBond(old_to_new[anchor_idx], dummy_idx, Chem.BondType.SINGLE)

    for bond in macro_mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        if a1 in old_to_new and a2 in old_to_new:
            emol.AddBond(old_to_new[a1], old_to_new[a2], bond.GetBondType())

    linker = emol.GetMol()
    try:
        Chem.SanitizeMol(linker)
        new_conf = Chem.Conformer(linker.GetNumAtoms())
        for old_idx, new_idx in old_to_new.items():
            pos = conf.GetAtomPosition(old_idx)
            new_conf.SetAtomPosition(new_idx, pos)
        for anchor_idx, dummy_idx in zip(selected_anchors, dummy_indices):
            pos = conf.GetAtomPosition(anchor_idx)
            new_conf.SetAtomPosition(dummy_idx, pos)
        linker.AddConformer(new_conf)
    except Exception as e:
        print(f"⚠  Could not sanitize linker or set coordinates: {e}", file=sys.stderr)
        return None
    return linker

def mol_to_unique_smiles(mol):
    try:
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    except:
        return None

def find_terminal_atoms(mol):
    # Find two terminal atoms with available valence from both ends of the linker.
    atoms = mol.GetAtoms()
    n = len(atoms)

    left_idx = None
    right_idx = None

    # Left-to-right: find first atom with available valence
    for i in range(n):
        atom = atoms[i]
        if atom.GetAtomicNum() == 0:
            continue
        if atom.GetTotalNumHs() > 0:
            left_idx = i
            break

    # Right-to-left: find first atom with available valence
    for i in range(n - 1, -1, -1):
        atom = atoms[i]
        if atom.GetAtomicNum() == 0:
            continue
        if atom.GetTotalNumHs() > 0:
            right_idx = i
            break

    if left_idx is None or right_idx is None or left_idx == right_idx:
        print("❌ Could not find 2 suitable anchor atoms with available valence")
        return []

    print("TERMINAL (shifted): ⭕⭕", [left_idx, right_idx])
    return [left_idx, right_idx]



def read_valid_smiles(smiles_file):
    valid_smiles = []
    with open(smiles_file, 'r') as f:
        for line in f:
            smi = line.strip().replace(" ","")
            mol = Chem.MolFromSmiles(smi)
            if mol is not None and smi.count('*') == 2:
                valid_smiles.append(smi)

    return valid_smiles

from rdkit import Chem

def read_linkers_from_sdf(sdf_file):
    suppl = Chem.SDMolSupplier(sdf_file, sanitize=True, removeHs=False)
    linkers = []
    for mol in suppl:
        if mol is None:
            print("❌ Invalid mol: None")
            continue

        smiles = Chem.MolToSmiles(mol)

        # check for 2 dummy atoms
        if smiles.count('*') != 2:
            print(f"❌ Invalid linker (must contain 2 * atoms): {smiles}")
            continue

        # check for single fragment
        frags = Chem.GetMolFrags(mol, asMols=True)
        if len(frags) > 1:
            print(f"❌ Invalid linker (disconnected fragments): {smiles}")
            continue

        # ✅ passed all checks
        linkers.append(smiles)

    return linkers

from rdkit import Chem

def move_star_if_needed(mol):
    
    # If * atom is connected to an atom whose valency is already full,
    # try to move the * to a neighboring atom with available valence.
    
    stars = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]  # 0 = Dummy atom (*)
    for star in stars:
        neighbor = star.GetNeighbors()[0] if star.GetNeighbors() else None
        if neighbor:
            max_valence = Chem.GetPeriodicTable().GetDefaultValence(neighbor.GetAtomicNum())
            if neighbor.GetTotalValence() >= max_valence:
                # Try shifting * to another neighbor
                for alt in neighbor.GetNeighbors():
                    if alt.GetIdx() != star.GetIdx():
                        alt_max_valence = Chem.GetPeriodicTable().GetDefaultValence(alt.GetAtomicNum())
                        if alt.GetTotalValence() < alt_max_valence:
                            # Move star: remove old *, add new one on alt
                            emol = Chem.RWMol(mol)
                            emol.RemoveAtom(star.GetIdx())
                            new_star = emol.AddAtom(Chem.Atom(0))
                            emol.AddBond(alt.GetIdx(), new_star, Chem.rdchem.BondType.SINGLE)
                            mol = emol.GetMol()
                            Chem.SanitizeMol(mol)
                            return mol
    return mol


def connect_fragment_and_linker(fragment_smi, linker_smi):
    print(fragment_smi)
    print(linker_smi)
    frag = Chem.MolFromSmiles(fragment_smi)
    linker = Chem.MolFromSmiles(linker_smi)

    if frag is None or linker is None:
        print("Invalid fragment or linker SMILES")
        return None

    # Adjust * positions if needed
    frag = move_star_if_needed(frag)
    linker = move_star_if_needed(linker)

    # Find * atoms in both
    frag_stars = [a.GetIdx() for a in frag.GetAtoms() if a.GetAtomicNum() == 0]
    linker_stars = [a.GetIdx() for a in linker.GetAtoms() if a.GetAtomicNum() == 0]

    if len(frag_stars) != 2 or len(linker_stars) != 2:
        raise ValueError(f"Expected exactly 2 '*' atoms in fragment and linker, found {len(frag_stars)} and {len(linker_stars)}")

    # Combine molecules
    combo = Chem.CombineMols(frag, linker)
    linker_offset = frag.GetNumAtoms()

    # Map linker star indices to combo indices
    linker_stars = [idx + linker_offset for idx in linker_stars]

    # Get anchor atoms (neighbors of the stars)
    frag_anchors = [combo.GetAtomWithIdx(idx).GetNeighbors()[0].GetIdx() for idx in frag_stars]
    linker_anchors = [combo.GetAtomWithIdx(idx).GetNeighbors()[0].GetIdx() for idx in linker_stars]

    # Build editable molecule
    emol = Chem.EditableMol(combo)

    # Connect fragment anchors to linker anchors
    emol.AddBond(frag_anchors[0], linker_anchors[0], Chem.rdchem.BondType.SINGLE)
    emol.AddBond(frag_anchors[1], linker_anchors[1], Chem.rdchem.BondType.SINGLE)

    # Remove * atoms (in reverse order)
    for idx in sorted(frag_stars + linker_stars, reverse=True):
        emol.RemoveAtom(idx)

    result = emol.GetMol()
    Chem.SanitizeMol(result)
    return result


def is_macrocycle(smiles, min_ring_size=11):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    ring_info = mol.GetRingInfo()
    return any(len(ring) >= min_ring_size for ring in ring_info.AtomRings())

def process_ops(input_file, output_file):
    macrocycles = set()

    with open(input_file, 'r') as f:
        for line in f:
            smiles = line.strip()
            if not smiles:
                continue
            if is_macrocycle(smiles):
                macrocycles.add(smiles)

    with open(output_file, 'a') as out:
        for smi in sorted(macrocycles):
            out.write(smi + '\n')

    print(f"✅ Found {len(macrocycles)} unique macrocycles.")
    print(f"📄 Written to: {output_file}")

def calculate_physicochemical_properties(smiles):
    # Calculate physicochemical properties for a given SMILES string.
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
        synthetic_accessibility=calculate_sa_score(mol)
        
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
    # Predict simplified ADMET properties based on physicochemical properties.
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    psa = MolSurf.TPSA(mol)
    
    # Simplified permeability prediction (based on LogP and PSA)
    permeability = 'High' if logp > 2.5 and psa < 250 else 'Low'
    
    # Simplified hERG liability (lipophilic and basic amines increase risk)
    hba = Lipinski.NumHAcceptors(mol)
    hbd = Lipinski.NumHDonors(mol)
    herg_risk = 'High' if logp > 3 and hba > 3 else 'Low'
    
    # Simplified CYP inhibition (based on molecular complexity)
    cyp_risk = 'Low' if mw < 800 and hba < 10 else 'Moderate'
    
    # Simplified metabolic stability (fewer heteroatoms = more stable)
    num_heteroatoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in [6, 1])
    stability = 'High' if num_heteroatoms < 10 else 'Moderate'
    
    # PAINS-like filter for reactive groups
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
        'lipinski_violations':lipinski,
        'QED': QED.qed(mol)
    }

def calculate_druglikeness_score(props, admet):
   # Calculate a drug-likeness score based on physicochemical and ADMET properties.
    score = 0
    
    # Physicochemical scoring
    if props['MW'] < 600:
        score += 10
    if props['LogP'] >= 0.5 and props['LogP']<=5:
        score += 20
    if props['HBD'] <= 5:
        score += 20
    if props['HBA'] <= 10:
        score += 20
    if props['PSA'] < 140:
        score += 20
    if props['RotatableBonds'] < 10:
        score += 10
    if props['SA'] is not None and 1 <= props['SA'] <= 3:
        score += 30
    
    # ADMET scoring
    if admet['Permeability'] == 'High':
        score += 10
    if admet['hERG_Risk'] == 'Low':
        score += 15
    if admet['CYP_Inhibition'] == 'Low':
        score += 10
    if admet['Metabolic_Stability'] == 'High':
        score += 10
    if admet['PAINS_Alert'] == 'No':
        score += 20
    if admet['QED']>0.5:
        score+=30
    
    return score

def filter_macrocycles(input_file, output_file, input_smiles=None, macro_linker_pairs=None):
    # Read SMILES from input file (each line is a SMILES string)
    with open(input_file, 'r') as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    
    # Initialize lists to store results
    results = []
    
    # Create a dictionary to map macrocycle SMILES to linker SMILES
    linker_map = {macro_smi: linker_smi for macro_smi, linker_smi in macro_linker_pairs} if macro_linker_pairs else {}

    # Calculate properties for input SMILES (no filtering)
    if input_smiles:
        props = calculate_physicochemical_properties(input_smiles)
        if props:
            mol = Chem.MolFromSmiles(input_smiles)
            if mol:
                admet = predict_admet_properties(mol)
                score = calculate_druglikeness_score(props, admet)
                input_result = {**props, **admet, 'Druglikeness_Score': score, 'Linkers': None}  # No linker for input
                results.append(input_result)
    
    # Calculate properties for output macrocycles
    for smiles in smiles_list:
        props = calculate_physicochemical_properties(smiles)
        if props is None:
            continue
        print("YESS\n\n" + smiles)
        mol = Chem.MolFromSmiles(smiles)
        admet = predict_admet_properties(mol)
        score = calculate_druglikeness_score(props, admet)
        # Add linker SMILES if available
        linker_smi = linker_map.get(smiles, None)  # Get linker for this macrocycle
        result = {**props, **admet, 'Druglikeness_Score': score, 'Linkers': linker_smi}
        results.append(result)
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Separate input SMILES (if any) and output macrocycles
    if input_smiles and not results_df.empty:
        input_df = results_df[results_df['SMILES'] == input_smiles]
        output_df = results_df[results_df['SMILES'] != input_smiles]
    else:
        input_df = pd.DataFrame()
        output_df = results_df
    
    print("#####################################################\n\n" + str(output_df.shape))
    print(output_df.head(10))
    
    # Take top 5 based on drug-likeness score only (no filters)
    if output_df.shape[0] <= 5:
        print(str(output_df.shape) + "########\n\n")
        print("😭 Less than or equal to 5 macrocycles")
        top_5_df = output_df
    else:
        top_5_df = output_df.sort_values(by='Druglikeness_Score', ascending=False).head(5)
    
    # Combine input SMILES (if any) with top 5 macrocycles
    final_df = pd.concat([input_df, top_5_df], ignore_index=True)
    
    # Save to output CSV
    final_df.to_csv(output_file, index=False)
    print(f"Top 5 macrocycles saved to {output_file}")
    print(f"Top 5 candidates (plus input SMILES if provided):\n{final_df.to_string()}")
    print(f"Number of macrocycles in final output: {len(top_5_df)}")
    
    # Return the top macrocycle (highest Druglikeness_Score) if any macrocycles exist
    if not output_df.empty:
        top_macrocycle = output_df.sort_values(by='Druglikeness_Score', ascending=False).iloc[0]
        return top_macrocycle
    return None

def process_single_smiles(user_smiles, inter_dir, input_id):
    # Make a dedicated folder for this input's run
    run_dir = f"{inter_dir}/input_{input_id}"
    os.makedirs(run_dir, exist_ok=True)

    user_sdf_path = os.path.join(run_dir, f"user_input.sdf")
    user_smiles = remove_radicals(user_smiles)
    smi_to_sdf(user_smiles, user_sdf_path)

    augmented = generate_augmented_smiles(user_smiles, N_AUGMENT)

    # Step 2: Run Macformer
    all_macformer_outputs = []
    for i, aug_smi in enumerate(augmented):
        try:
            outputs = run_macformer_on_smiles_in_memory(aug_smi, run_dir)
            all_macformer_outputs.extend(outputs)
        except subprocess.CalledProcessError as e:
            print(f"⚠ Macformer error on SMILES [{i}]: {e}")

    # Save predictions
    out_path = os.path.join(run_dir, f"valid_macformer_smiles.txt")
    with open(out_path, "a") as f:
        # 1️⃣ Write the original SMILES first
        f.write(user_smiles + "\n")
        # 2️⃣ Then write all predictions
        for pred in all_macformer_outputs:
            f.write(pred + "\n")

    # Step 3: Filter valid linkers
    valid_linker_smiles = filter_valid_smiles(all_macformer_outputs)
    
    with open(os.path.join(run_dir, f"cyclizer_smiles.txt"), "a") as f:
        for smiles in valid_linker_smiles:
            f.write(smiles + "\n")

    # Step 5: Run EDM with retries for empty linkers.sdf or no macro_linker_pairs
    max_retries = 5
    retry_count = 0
    linkers_sdf_path = os.path.join(run_dir, "linkers.sdf")
    valid_linkers_found = False
    macro_linker_pairs = []

    while retry_count < max_retries and (not macro_linker_pairs or top_macrocycle is None):
        try:
            # Run DiffLinker
            run_difflinker(run_dir)

            # Step 6: Extract linkers from this input's DiffLinker output
            acyclic_mol = load_mol(user_sdf_path)
            linker_writer = SDWriter(linkers_sdf_path)

            out_dir = os.path.join(run_dir, f"diff_linker_ops")
            for file in os.listdir(out_dir):
                if file.endswith(".sdf"):
                    macro_path = os.path.join(out_dir, file)
                    macro_mol = load_mol(macro_path)
                    if macro_mol:
                        linker_indices = find_linker_atom_indices(acyclic_mol, macro_mol)
                        if linker_indices:
                            linker_mol = extract_linker_from_macro(macro_mol, linker_indices)
                            if linker_mol:
                                linker_writer.write(linker_mol)

            linker_writer.close()
            print(f"✅ Linkers saved for input {input_id} in {run_dir}/linkers.sdf")

            # Check if linkers.sdf is empty or invalid
            try:
                suppl = Chem.SDMolSupplier(linkers_sdf_path, sanitize=False, removeHs=False)
                if any(mol is not None for mol in suppl):
                    valid_linkers_found = True
                else:
                    print(f"⚠ linkers.sdf is empty or contains no valid molecules for input {input_id}, retrying ({retry_count + 1}/{max_retries})...")
                    retry_count += 1
                    continue
            except Exception as e:
                print(f"⚠ Failed to read linkers.sdf: {e}, retrying ({retry_count + 1}/{max_retries})...")
                retry_count += 1
                continue

            # Step 7: Process linkers from SDF, split multi-fragment linkers, and ensure 2 * atoms
            def process_linker_mol(mol):
                if mol is None:
                    print("⚠ Skipping invalid molecule: None")
                    return []
                # Split into fragments
                frags = Chem.GetMolFrags(mol, asMols=True)
                processed_linkers = []
                for frag in frags:
                    try:
                        # Ensure molecule is sanitized
                        Chem.SanitizeMol(frag, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
                        frag_rw = Chem.RWMol(frag)
                        # Count existing * atoms
                        dummy_count = sum(1 for atom in frag_rw.GetAtoms() if atom.GetAtomicNum() == 0)
                        dummy_indices = [atom.GetIdx() for atom in frag_rw.GetAtoms() if atom.GetAtomicNum() == 0]
                        
                        # Get terminal atoms with free valence
                        terminal_indices = find_terminal_atoms(frag_rw)
                        if not terminal_indices:
                            print(f"⚠ Skipping fragment: no suitable terminal atoms with free valence")
                            continue
                        
                        if dummy_count == 0:
                            # Add two * atoms at terminal atoms
                            for idx in terminal_indices[:2]:  # Use first two terminal atoms
                                new_dummy = frag_rw.AddAtom(Chem.Atom(0))
                                frag_rw.AddBond(idx, new_dummy, Chem.BondType.SINGLE)
                        elif dummy_count == 1:
                            # Add one more * atom at a terminal atom with free valence
                            target_idx = [idx for idx in terminal_indices if idx not in dummy_indices]
                            if not target_idx:
                                print(f"⚠ Skipping fragment: no valid atom to add second *")
                                continue
                            new_dummy = frag_rw.AddAtom(Chem.Atom(0))
                            frag_rw.AddBond(target_idx[0], new_dummy, Chem.BondType.SINGLE)
                        elif dummy_count > 2:
                            # Keep only the first two * atoms
                            for idx in sorted(dummy_indices[2:], reverse=True):
                                frag_rw.RemoveAtom(idx)
                        # Ensure exactly 2 * atoms
                        dummy_indices = [atom.GetIdx() for atom in frag_rw.GetAtoms() if atom.GetAtomicNum() == 0]
                        if len(dummy_indices) != 2:
                            # Try adding * atoms using add_second_attachment_point_and_return_indices
                            new_dummy_indices = add_second_attachment_point_and_return_indices(frag_rw)
                            if new_dummy_indices is None or len(new_dummy_indices) != 2:
                                print(f"⚠ Skipping fragment: could not ensure exactly 2 * atoms")
                                continue
                            dummy_indices = new_dummy_indices
                        
                        # Final sanitization
                        try:
                            Chem.SanitizeMol(frag_rw)
                            linker_smi = Chem.MolToSmiles(frag_rw, canonical=True)
                            if linker_smi.count('*') == 2:
                                processed_linkers.append(linker_smi)
                                print(f"✅ Processed fragment: {linker_smi}")
                            else:
                                print(f"⚠ Skipping fragment: incorrect number of * atoms: {linker_smi}")
                        except Exception as e:
                            print(f"⚠ Skipping fragment: sanitization failed: {e}")
                    except Exception as e:
                        print(f"⚠ Skipping fragment: initial sanitization failed: {e}")
                return processed_linkers

            # Read and process linkers from SDF
            suppl = Chem.SDMolSupplier(linkers_sdf_path, sanitize=False, removeHs=False)
            linkers = []
            for mol in suppl:
                linkers.extend(process_linker_mol(mol))

            if not linkers:
                print(f"⚠ No valid linkers after processing for input {input_id}, retrying ({retry_count + 1}/{max_retries})...")
                retry_count += 1
                continue

            with open(f"{inter_dir}/input_{input_id}/extracted_linkers.txt", 'a') as f:
                for linker in linkers:
                    f.write(linker + '\n')
            print(f"✅ Extracted linkers SMILES saved to {inter_dir}/input_{input_id}/extracted_linkers.txt")

            valid_smiles = read_valid_smiles(f"{inter_dir}/input_{input_id}/cyclizer_smiles.txt")

            # Skip if cyclizer_smiles.txt is empty
            if not valid_smiles:
                print(f"⚠ Skipping input {input_id} (no valid cyclizer smiles), retrying ({retry_count + 1}/{max_retries})...")
                retry_count += 1
                continue

            # Step 8: Generate macrocycles and track linkers
            macro_linker_pairs = []  # Store (macrocycle SMILES, linker SMILES) pairs
            for smi in list(set(valid_smiles)):
                for linker in linkers:
                    result = connect_fragment_and_linker(smi.replace(" ", ""), linker)
                    if result is not None:
                        result_smi = Chem.MolToSmiles(Chem.RemoveHs(result))
                        if result_smi:
                            macro_linker_pairs.append((result_smi, linker))
                            print(f"Generated macrocycle: {result_smi} with linker: {linker}")

            if not macro_linker_pairs:
                print(f"⚠ No valid macrocycles generated for input {input_id}, retrying ({retry_count + 1}/{max_retries})...")
                retry_count += 1
                continue

            # Write macrocycle SMILES to ops.txt
            with open(f"{inter_dir}/input_{input_id}/ops.txt", 'a') as f_out:
                for macro_smi, _ in macro_linker_pairs:
                    f_out.write(macro_smi + '\n')
        
            process_ops(f"{inter_dir}/input_{input_id}/ops.txt", f"{inter_dir}/input_{input_id}/final_macros.txt")
            print(f"Macrocycles generated successfully")
        
            # Pass macro_linker_pairs to filter_macrocycles
            top_macrocycle = filter_macrocycles(
                f"{inter_dir}/input_{input_id}/final_macros.txt",
                f"{inter_dir}/input_{input_id}/top_5_macrors.csv",
                input_smiles=user_smiles,
                macro_linker_pairs=macro_linker_pairs
            )
            if top_macrocycle is None:
                print(f"⚠ Filter removed all macros for input {input_id}, retrying ({retry_count + 1}/{max_retries})...")
                retry_count += 1
                continue

            print("✅ Macrocycles generated successfully")

        except Exception as e:
            print(f"⚠ Error during processing: {e}, retrying ({retry_count + 1}/{max_retries})...")
            retry_count += 1

    if top_macrocycle is None:
        print(f"⚠ Skipping input {input_id} (no valid macrocycles after {max_retries} retries)")
        return None

    return {'Input_SMILES': user_smiles, 'Top_Macrocycle': top_macrocycle}    
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,7"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    parser = argparse.ArgumentParser(description="Process multiple SMILES from a file.")
    parser.add_argument("--input_file", default="inference_test.txt", help="Path to the file containing SMILES strings (one per line)")
    args = parser.parse_args()

    # List to store top macrocycles for each input
    top_macrocycles = []

    with open(args.input_file, 'r') as f:
        smiles_list = [line.strip() for line in f if line.strip()]

    for i, user_smiles in enumerate(smiles_list):
        inter_dir = INTER_DIR
        print(f"Processing SMILES {i}/{len(smiles_list)}: {user_smiles}")
        result = process_single_smiles(user_smiles, inter_dir, i)
        if result is not None and result.get('Top_Macrocycle') is not None:
            # Add input SMILES and linker to the top macrocycle's properties
            top_macrocycle = result['Top_Macrocycle'].to_dict()
            top_macrocycle['Input_SMILES'] = result['Input_SMILES']
            top_macrocycles.append(top_macrocycle)

    # Create DataFrame for top macrocycles
    if top_macrocycles:
        top_macrocycles_df = pd.DataFrame(top_macrocycles)
        # Reorder columns to put Input_SMILES and Linkers first
        cols = ['Input_SMILES', 'SMILES', 'Linkers'] + [col for col in top_macrocycles_df.columns if col not in ['Input_SMILES', 'SMILES', 'Linkers']]
        top_macrocycles_df = top_macrocycles_df[cols]
        # Save to CSV
        output_csv = os.path.join(INTER_DIR, "top_performing_per_input.csv")
        top_macrocycles_df.to_csv(output_csv, index=False)
        print(f"✅ Top performing macrocycles for each input saved to {output_csv}")
    else:
        print("⚠ No valid macrocycles found for any input SMILES")
