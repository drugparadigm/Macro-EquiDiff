#!/usr/bin/env python

import argparse
from rdkit import Chem

def get_largest_fragment(mol):
    """Returns the largest fragment from a potentially disconnected molecule"""
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if not frags:
        return None
    return max(frags, key=lambda m: m.GetNumAtoms())

def connect_fragment_and_linker(fragment_smi, linker_smi):
    # Load and sanitize fragment
    frag = Chem.MolFromSmiles(fragment_smi)
    if frag is None:
        raise ValueError("Invalid fragment SMILES")


    # Load linker, keep only the largest fragment
    linker_mol = Chem.MolFromSmiles(linker_smi)
    print(linker_smi)
    print("⭕")
    if linker_mol is None:
        raise ValueError("Invalid linker SMILES")

    linker = get_largest_fragment(linker_mol)
    if linker is None:
        raise ValueError("Failed to extract valid linker fragment")

    # Combine both
    combo = Chem.CombineMols(frag, linker)
    
    # Find all dummy atoms
    dummy_indices = [atom.GetIdx() for atom in combo.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummy_indices) != 4:
        raise ValueError(f"Expected 4 dummy atoms (2 in fragment, 2 in linker), found {len(dummy_indices)}")

    # First two are from frag, next two from linker
    frag_dummies = dummy_indices[:2]
    linker_dummies = dummy_indices[2:]

    # Get anchor atoms (neighbor to dummy)
    frag_anchors = [list(combo.GetAtomWithIdx(idx).GetNeighbors())[0].GetIdx() for idx in frag_dummies]
    linker_anchors = [list(combo.GetAtomWithIdx(idx).GetNeighbors())[0].GetIdx() for idx in linker_dummies]

    # Build editable molecule
    emol = Chem.EditableMol(combo)
    emol.AddBond(frag_anchors[0], linker_anchors[0], Chem.rdchem.BondType.SINGLE)
    emol.AddBond(frag_anchors[1], linker_anchors[1], Chem.rdchem.BondType.SINGLE)

    # Remove dummy atoms (in reverse order)
    for idx in sorted(dummy_indices, reverse=True):
        emol.RemoveAtom(idx)

    result = emol.GetMol()
    Chem.SanitizeMol(result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Attach linker SMILES to fragment SMILES at dummy atoms ('[*]').")
    parser.add_argument('--fragment', required=True, help="Fragment SMILES with two [*] anchor points")
    parser.add_argument('--linker', required=True, help="Linker SMILES with two [*] endpoints")
    parser.add_argument('--output', default=None, help="Optional file path to write the final SMILES")
    args = parser.parse_args()

    try:
        final_mol = connect_fragment_and_linker(args.fragment, args.linker)
        final_smi = Chem.MolToSmiles(Chem.RemoveHs(final_mol))
        print(final_smi)
        if args.output:
            with open(args.output, 'w') as f:
                f.write(final_smi + '\n')
            print(f"✅ Final SMILES written to {args.output}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
