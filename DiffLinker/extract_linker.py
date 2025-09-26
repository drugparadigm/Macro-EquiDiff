import os
import argparse
import sys
from rdkit import Chem
from rdkit.Chem import AllChem, SDWriter
import numpy as np
from rdkit.Chem import rdmolops

def load_mol(sdf_path):
    try:
        suppl = Chem.SDMolSupplier(sdf_path, sanitize=True, removeHs=False)
        for mol in suppl:
            if mol is not None:
                return mol
    except Exception as e:
        print(f"⚠️  RDKit load error for {sdf_path}: {e}", file=sys.stderr)
    print(f"⚠️  Could not load molecule from {sdf_path} (skipped)", file=sys.stderr)
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
    """Find linker atoms by removing the number of heavy atoms in acyclic_mol from macro_mol."""
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
    """Find linker atoms that were connected to non-linker (acyclic) atoms."""
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
    return dummy_idxs


def select_farthest_anchor_points(macro_mol, anchor_points):
    """Select the two anchor points that are farthest apart, preferring non-ring atoms."""
    if len(anchor_points) <= 2:
        return anchor_points

    ring_info = macro_mol.GetRingInfo()
    non_ring_anchors = [idx for idx in anchor_points if not ring_info.IsAtomInRingOfSize(idx, ring_info.NumRings())]
    ring_anchors = [idx for idx in anchor_points if ring_info.IsAtomInRingOfSize(idx, ring_info.NumRings())]

    conf = macro_mol.GetConformer()
    max_dist = -1
    farthest_pair = None

    if len(non_ring_anchors) >= 2:
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
        print(f"⚠️  Fewer than 2 non-ring anchor points ({len(non_ring_anchors)}), using ring atoms if necessary", file=sys.stderr)
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
        print(f"⚠️  Could not find valid anchor point pair", file=sys.stderr)
        return None
    return farthest_pair

def extract_linker_from_macro(macro_mol, linker_atom_indices):
    anchor_points = find_anchor_points(macro_mol, linker_atom_indices)
    print("🧠",anchor_points)
    if anchor_points is None or len(anchor_points) != 2:
        print(f"⭕  Expected exactly 2 anchor points, found {len(anchor_points)} after selection", file=sys.stderr)

        # Try to fix: Add second dummy if only one present
        if anchor_points is not None and len(anchor_points) == 1:
            print("🛠️  Attempting to add second attachment point...")
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
        print(f"⚠️  Expected exactly 2 anchor points, found {len(selected_anchors)} after selection", file=sys.stderr)
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
        print(f"⚠️  Could not sanitize linker or set coordinates: {e}", file=sys.stderr)
        return None
    return linker

def mol_to_unique_smiles(mol):
    try:
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    except:
        return None

def main():
    parser = argparse.ArgumentParser(description="Extract linkers from all macrocycle SDFs in a folder by removing the number of heavy atoms in the acyclic fragment, adding dummy atoms (*) to farthest non-ring anchor points, and store unique, valid, non-fragment linkers in an output SDF.")
    parser.add_argument("--acyclic", default="intermediates/raw_input.sdf",help="Path to acyclic fragment SDF")
    parser.add_argument("--macro", required=True, help="Path to the single macrocycle SDF file")
    parser.add_argument("--output",default="intermediates/all_linkers.sdf", help="Output SDF path for unique extracted linkers")

    args = parser.parse_args()


    # Load acyclic molecule
    acyclic_mol = load_mol(args.acyclic)
    if acyclic_mol is None:
        print(f"❌ Could not load acyclic file: {args.acyclic} -- exiting", file=sys.stderr)
        sys.exit(1)

    # Initialize writer for unique linkers
    writer = SDWriter(args.output)
    unique_smiles = set()
    sdf_count = 0
    linker_count = 0

    # Iterate through all SDF files in the macro_folder
    # Validate macro file
    if not os.path.isfile(args.macro):
        print(f"❌ Macrocycle SDF does not exist: {args.macro}", file=sys.stderr)
        sys.exit(1)
    
    macro_path = args.macro

    sdf_count += 1
    print(f"📄 Processing: {macro_path}", file=sys.stderr)
    macro_mol = load_mol(macro_path)
    if macro_mol is None:
        print(f"⚠️  Could not load macrocycle file: {macro_path} -- skipping", file=sys.stderr)
        return

    # Extract linker
    linker_indices = find_linker_atom_indices(acyclic_mol, macro_mol)
    if not linker_indices:
        print(f"⚠️  No linker atoms found for {macro_path}.", file=sys.stderr)
        return

    linker_mol = extract_linker_from_macro(macro_mol, linker_indices)
    if linker_mol is None:
        print(f"⚠️  Failed to build valid linker from {macro_path}.", file=sys.stderr)
        return

    # Check for validity: must have exactly 2 dummy atoms (*)
    dummy_count = sum(1 for atom in linker_mol.GetAtoms() if atom.GetAtomicNum() == 0)
    if dummy_count != 2:
        print(f"⚠️  Linker from {macro_path} has {dummy_count} dummy atoms, expected 2 -- skipping", file=sys.stderr)
        return

    # Check for non-fragment: must be a single connected molecule
    fragments = Chem.GetMolFrags(linker_mol)
    if len(fragments) > 1:
        print(f"⚠️  Linker from {macro_path} has {len(fragments)} fragments, expected 1 -- skipping", file=sys.stderr)
        return

    # Check for uniqueness
    linker_smiles = mol_to_unique_smiles(linker_mol)
    if linker_smiles and linker_smiles not in unique_smiles:
        unique_smiles.add(linker_smiles)
        writer.write(linker_mol)
        linker_count += 1
        print(f"✅ Linker extracted from {macro_path} and added to {args.output}")

    writer.close()
    print(f"🏁 Processed {sdf_count} SDF files, extracted {linker_count} unique, valid, non-fragment linkers to {args.output}")

if __name__ == "__main__":
    main()