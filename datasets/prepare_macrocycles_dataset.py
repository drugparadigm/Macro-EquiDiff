import argparse
import itertools
import numpy as np
import pandas as pd

from itertools import product
from rdkit import Chem, Geometry
from tqdm import tqdm

import signal

class TimeoutException(Exception): pass

def handler(signum, frame):
    raise TimeoutException()

def get_exits(mol):
    exits = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol == '*':
            exits.append(atom)
    return exits

def set_anchor_flags(mol, anchor_idx):
    for atom in mol.GetAtoms():
        if atom.GetIdx() == anchor_idx:
            atom.SetProp('_Anchor', '1')
        else:
            atom.SetProp('_Anchor', '0')

def get_anchors_idx(mol):
    anchors_idx = []
    for atom in mol.GetAtoms():
        if atom.GetProp('_Anchor') == '1':
            anchors_idx.append(atom.GetIdx())
    return anchors_idx

def update_fragment(frag):
    exits = get_exits(frag)
    exits = sorted(exits, key=lambda e: e.GetIdx(), reverse=True)
    # Set anchor labels
    for exit in exits:
        bonds = exit.GetBonds()
        if len(bonds) > 1:
            raise Exception('Exit atom has more than 1 bond')
        bond = bonds[0]
        exit_idx = exit.GetIdx()
        source_idx = bond.GetBeginAtomIdx()
        target_idx = bond.GetEndAtomIdx()
        anchor_idx = source_idx if target_idx == exit_idx else target_idx
        set_anchor_flags(frag, anchor_idx)
    efragment = Chem.EditableMol(frag)
    # Remove exit bonds
    for exit in exits:
        bonds = exit.GetBonds()
        if len(bonds) > 1:
            raise Exception('Exit atom has more than 1 bond')
        bond = bonds[0]
        source_idx = bond.GetBeginAtomIdx()
        target_idx = bond.GetEndAtomIdx()
        efragment.RemoveBond(source_idx, target_idx)
    # Remove exit atoms
    for exit in exits:
        efragment.RemoveAtom(exit.GetIdx())
    return efragment.GetMol()

def update_linker(linker):
    exits = get_exits(linker)
    exits = sorted(exits, key=lambda e: e.GetIdx(), reverse=True)
    elinker = Chem.EditableMol(linker)
    # Remove exit bonds
    for exit in exits:
        bonds = exit.GetBonds()
        if len(bonds) > 1:
            raise Exception('Exit atom has more than 1 bond')
        bond = bonds[0]
        source_idx = bond.GetBeginAtomIdx()
        target_idx = bond.GetEndAtomIdx()
        elinker.RemoveBond(source_idx, target_idx)
    # Remove exit atoms
    for exit in exits:
        elinker.RemoveAtom(exit.GetIdx())
    return elinker.GetMol()

def create_conformer(coords):
    conformer = Chem.Conformer()
    for i, (x, y, z) in enumerate(coords):
        conformer.SetAtomPosition(i, Geometry.Point3D(x, y, z))
    return conformer

def transfer_conformers(frag, mol):
    matches = mol.GetSubstructMatches(frag)
    if len(matches) < 1:
        raise Exception('Could not find fragment or linker matches')
    match2conf = {}
    for match in matches:
        mol_coords = mol.GetConformer().GetPositions()
        frag_coords = mol_coords[np.array(match)]
        frag_conformer = create_conformer(frag_coords)
        match2conf[match] = frag_conformer
    return match2conf

def find_non_intersecting_matches(matches):
    combinations = list(product(*matches))[:10000]
    non_intersecting_matches = set()
    for combination in combinations:
        all_idx = []
        for match in combination:
            all_idx += match
        if len(all_idx) == len(set(all_idx)):
            non_intersecting_matches.add(combination)
    return list(non_intersecting_matches)

def find_matches_with_linker_in_the_middle(non_intersecting_matches, num_frags, mol):
    matches_with_linker_in_the_middle = []
    for m in non_intersecting_matches:
        fragments_m = m[:num_frags]
        linkers_m = m[num_frags:]
        all_frag_atoms = set()
        for frag_match in fragments_m:
            all_frag_atoms |= set(frag_match)
        all_linkers_are_in_the_middle = True
        for linker_match in linkers_m:
            linker_neighbors = set()
            for atom_idx in linker_match:
                atom_neighbors = mol.GetAtomWithIdx(atom_idx).GetNeighbors()
                for neighbor in atom_neighbors:
                    linker_neighbors.add(neighbor.GetIdx())
            number_of_connections = len(linker_neighbors & all_frag_atoms)
            if number_of_connections < 2:
                all_linkers_are_in_the_middle = False
                break
        if all_linkers_are_in_the_middle:
            matches_with_linker_in_the_middle.append(m)
    return matches_with_linker_in_the_middle

def find_correct_match(list_of_match_frag, list_of_match_linker, mol):
    non_intersecting_matches = find_non_intersecting_matches(list_of_match_frag + list_of_match_linker)
    if len(non_intersecting_matches) == 1:
        frag_match = non_intersecting_matches[0][:len(list_of_match_frag)]
        link_match = non_intersecting_matches[0][len(list_of_match_frag):]
        return frag_match, link_match
    matches_with_linker_in_the_middle = find_matches_with_linker_in_the_middle(
        non_intersecting_matches=non_intersecting_matches,
        num_frags=len(list_of_match_frag),
        mol=mol,
    )
    frag_match = matches_with_linker_in_the_middle[0][:len(list_of_match_frag)]
    link_match = matches_with_linker_in_the_middle[0][len(list_of_match_frag):]
    return frag_match, link_match

def prepare_fragments_and_linker(frags_smi, linker_smi, molecule):
    frags = []
    linkers = []
    for smi in frags_smi.split('.'):
        try:
            frags.append(Chem.MolFromSmiles(smi))
        except Exception as e:
            print(f"Error in Chem.MolFromSmiles for fragment {smi}: {e}")
            raise
    for smi in linker_smi.split('.'):
        try:
            linkers.append(Chem.MolFromSmiles(smi))
        except Exception as e:
            print(f"Error in Chem.MolFromSmiles for linker {smi}: {e}")
            raise
    new_frags = []
    for mol in frags:
        try:
            new_frags.append(update_fragment(mol))
        except Exception as e:
            print(f"Error updating fragment: {e}")
            raise
    new_linkers = []
    for mol in linkers:
        try:
            new_linkers.append(update_linker(mol))
        except Exception as e:
            print(f"Error updating linker: {e}")
            raise

    list_of_match2conf_frag = []
    list_of_match_frag = []
    for frag in new_frags:
        try:
            match2conf_frag = transfer_conformers(frag, molecule)
        except Exception as e:
            print(f"Error transferring fragment conformers: {e}")
            raise
        list_of_match2conf_frag.append(match2conf_frag)
        list_of_match_frag.append(list(match2conf_frag.keys()))

    list_of_match2conf_linker = []
    list_of_match_linker = []
    for linker in new_linkers:
        try:
            match2conf_linker = transfer_conformers(linker, molecule)
        except Exception as e:
            print(f"Error transferring linker conformers: {e}")
            raise
        list_of_match2conf_linker.append(match2conf_linker)
        list_of_match_linker.append(list(match2conf_linker.keys()))

    frag_matches, link_matches = find_correct_match(list_of_match_frag, list_of_match_linker, molecule)

    final_frag_mols = []
    for frag, frag_match, match2conf_frag in zip(new_frags, frag_matches, list_of_match2conf_frag):
        conformer = match2conf_frag[frag_match]
        frag.AddConformer(conformer)
        final_frag_mols.append(frag)

    final_link_mols = []
    for link, link_match, match2conf_link in zip(new_linkers, link_matches, list_of_match2conf_linker):
        conformer = match2conf_link[link_match]
        link.AddConformer(conformer)
        final_link_mols.append(link)

    return final_frag_mols, final_link_mols

def process_sdf(sdf_path, table, progress=True, verbose=True):
    supplier = list(Chem.SDMolSupplier(sdf_path))
    molecules = []
    fragments = []
    linkers = []
    out_table = []
    uuid = 0

    supplier = tqdm(supplier, total=len(supplier)) if progress else supplier
    for mol_idx, mol in enumerate(supplier):
        if mol is None:
            print(f"Skipping empty molecule at index {mol_idx}")
            continue
        try:
            mol_name = mol.GetProp('_Name')
        except Exception:
            print(f"Error reading molecule at index {mol_idx}; skipping.")
            continue
        print(f"Processing molecule {mol_idx}/{len(supplier)}: {mol_name}", flush=True)
        mol_smi = Chem.MolToSmiles(mol)
        mol.SetProp('_Name', mol_smi)
        for linker_smi, frags_smi in table[table.molecule == mol_name][['linker', 'fragments']].values:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(20)  # 20 seconds timeout
            try:
                frags, linker = prepare_fragments_and_linker(frags_smi, linker_smi, mol)
            except TimeoutException:
                print(f"TIMEOUT: Skipping molecule {mol_idx} ({mol_name}) due to timeout for linker={linker_smi} frags={frags_smi}")
                continue
            except Exception as e:
                print(f"ERROR: Skipping molecule {mol_idx} ({mol_name}) due to error: {e}")
                continue
            finally:
                signal.alarm(0)

            combined_frag = None
            for frag in frags:
                if combined_frag is None:
                    combined_frag = frag
                else:
                    combined_frag = Chem.CombineMols(combined_frag, frag)

            # anchors_idx = get_anchors_idx(combined_frag)

            combined_link = None
            for link in linker:
                if combined_link is None:
                    combined_link = link
                else:
                    combined_link = Chem.CombineMols(combined_link, link)

            molecules.append(mol)
            fragments.append(combined_frag)
            linkers.append(combined_link)

            out_table.append({
                'uuid': uuid,
                'molecule': mol_smi,
                'fragments': Chem.MolToSmiles(combined_frag),
                'linker': Chem.MolToSmiles(combined_link),
                'energy': '0', #mol.GetProp('_Energy'),
            })
            # 'anchors': '-'.join(map(str, anchors_idx)), use this line if you have anchor context
            uuid += 1

    return molecules, fragments, linkers, pd.DataFrame(out_table)

def run(table_path, sdf_path, out_mol_path, out_frag_path, out_link_path, out_table_path, progress=True, verbose=True):
    print(f'Table will be saved to {out_table_path}')

    table = pd.read_csv(table_path)
    molecules, fragments, linkers, out_table = process_sdf(sdf_path, table, progress, verbose)

    out_table.to_csv(out_table_path, index=False)
    with Chem.SDWriter(open(out_mol_path, 'w')) as writer:
        for mol in molecules:
            writer.write(mol)
    with Chem.SDWriter(open(out_frag_path, 'w')) as writer:
        writer.SetKekulize(False)
        for frags in fragments:
            writer.write(frags)
    with Chem.SDWriter(open(out_link_path, 'w')) as writer:
        writer.SetKekulize(False)
        for linker in linkers:
            writer.write(linker)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--table', action='store', type=str, required=True)
    parser.add_argument('--sdf', action='store', type=str, required=True)
    parser.add_argument('--out-mol-sdf', action='store', type=str, required=True)
    parser.add_argument('--out-frag-sdf', action='store', type=str, required=True)
    parser.add_argument('--out-link-sdf', action='store', type=str, required=True)
    parser.add_argument('--out-table', action='store', type=str, required=True)
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()

    run(
        table_path=args.table,
        sdf_path=args.sdf,
        out_mol_path=args.out_mol_sdf,
        out_frag_path=args.out_frag_sdf,
        out_link_path=args.out_link_sdf,
        out_table_path=args.out_table,
        verbose=args.verbose,
    )