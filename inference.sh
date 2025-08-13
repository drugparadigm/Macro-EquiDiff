#!/usr/bin/env bash

set -e

echo ""
echo "============================================"
echo "🚀  Macrocycle Generation Full Pipeline Start"
echo "============================================"
echo ""

# # Create folders for intermediate files
# mkdir -p outputs
# mkdir -p intermediates

# #############################################
# # 1️⃣ User input
# #############################################
# echo "🔹 Please enter your raw acyclic SMILES string (without anchors):"
# read raw_smiles
# spaced_smiles=$(echo "$raw_smiles" | sed 's/./& /g' | sed 's/ $//')
# echo "$spaced_smiles" > intermediates/raw_input.smi
# echo "✅ Saved user input to intermediates/raw_input.smi"
# echo ""

# #############################################
# # 2️⃣ Phase 1 - Macformer: Generate acyclic with anchors
# #############################################
# # echo "🔹 [Phase 1] Adding cyclization points (Macformer env)"
# # conda run -n Macformer --no-capture-output \
# #   python ~/raid_macrocycles/Macformer/translate.py \
# #     -model ~/raid_macrocycles/Macformer/checkpoints/diff/diff_modell_step_167000.pt \
# #     -src intermediates/raw_input.smi \
# #     -output intermediates/acyclic_with_anchors.smi \
# #     -batch_size 32 \
# #     -replace_unk \
# #     -max_length 500 \
# #     -beam_size 10 \
# #     -n_best 10 \
# #     -gpu 2

# echo "✅ Phase 1 complete!"
# echo "   ➜ Generated acyclic with anchors at intermediates/raw_input.sdf.smi"
# echo ""

# #############################################
# # 1️⃣.5 NEW STEP: Convert .smi to .sdf
# #############################################
# echo "🔹 [Phase 1.5] Converting SMILES to SDF (RDKit env)"
# conda run -n dl --no-capture-output python -c "
# from rdkit import Chem
# from rdkit.Chem import AllChem
# with open('intermediates/raw_input.smi') as f:
#     smiles = f.readline().replace(' ','')
# mol = Chem.MolFromSmiles(smiles)
# mol = Chem.AddHs(mol)
# AllChem.EmbedMolecule(mol)
# writer = Chem.SDWriter('intermediates/raw_input.sdf')
# writer.write(mol)
# writer.close()
# "
# echo "✅ SMILES converted to intermediates/acyclic_with_anchors.sdf"
# echo ""

# #############################################
# # 3️⃣ Phase 2 - dl: Generate macrocycle structure
# #############################################
# echo "🔹 [Phase 2] Generating macrocycle (dl env)"
# conda run -n dl --no-capture-output \
#   python -W ignore ~/DiffLinker/generate.py \
#     --fragments intermediates/raw_input.sdf \
#     --model ~/DiffLinker/models/geom_difflinker.ckpt \
#     --linker_size "6"\
#     --output intermediates/phase2_op

# echo "✅ Phase 2 complete!"
# echo "   ➜ Macrocycle output saved in SDF (e.g. intermediates/phase2_op/*.sdf)"
# echo ""

#############################################
# 3️⃣ Phase 3 - Extract Linkers from All SDFs
#############################################
echo "🔹 [Phase 3] Extracting linkers from macrocycle SDF files"

# Clear old linkers
> intermediates/all_linkers.sdf

# Iterate over all SDF files in phase2_op
for sdf_file in intermediates/phase2_op/*.sdf; do
    if [ -s "$sdf_file" ]; then  # check if not empty
        echo "🔍 Processing: $sdf_file"

        # Try to extract linker
        conda run -n dl --no-capture-output \
          python ~/DiffLinker/extract_linker.py \
            --acyclic intermediates/raw_input.sdf \
            --macro "$sdf_file" \
            --output intermediates/tmp_linker.sdf

        # If successful, append to all_linkers.sdf
        if [ -s intermediates/tmp_linker.sdf ]; then
            echo "   ✅ Linker extracted from $sdf_file"
            cat intermediates/tmp_linker.sdf >> intermediates/all_linkers.sdf
        else
            echo "   ❌ Failed to extract from $sdf_file"
        fi
    fi
done
echo "✅ Linker extraction complete!"
echo "   ➜ Combined linkers saved to intermediates/all_linkers.sdf"
echo ""

#############################################
# 4️⃣ Phase 4 - Join ALL linkers to ALL fragments
#############################################
echo "🔹 [Phase 4] Attaching ALL linkers to ALL fragments"

# Extract all linkers as SMILES
echo "📦 Converting all linkers to SMILES"
linker_smi_list=$(conda run -n dl --no-capture-output python -c "
from rdkit import Chem
suppl = Chem.SDMolSupplier('intermediates/all_linkers.sdf')
for mol in suppl:
    if mol:
        print(Chem.MolToSmiles(mol))
")

# Prepare output file
> intermediates/final_macrocycles.smi

# Loop over fragments
while IFS= read -r line; do
    fragment_smi=$(echo "$line" | tr -d ' ')
    echo "🔹 Fragment: $fragment_smi"

    # Pre-check: is fragment chemically valid and anchorable?
    is_fragment_valid=$(conda run -n dl --no-capture-output python -c "
from rdkit import Chem
fragment_smi = '''$fragment_smi'''  # Pass SMILES from Bash
try:
    mol = Chem.MolFromSmiles(fragment_smi)
    if mol is None:
        raise ValueError('Invalid SMILES')

    # Check there are exactly 2 dummy atoms
    dummies = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) != 2:
        raise ValueError('Expected 2 dummy atoms')

    for dummy in dummies:
        neighbors = dummy.GetNeighbors()
        if len(neighbors) != 1:
            raise ValueError('Dummy not singly bonded')

        anchor = neighbors[0]

        # Count only heavy (non-H) neighbors (excluding the dummy itself)
        heavy_bond_count = sum(
            1 for nbr in anchor.GetNeighbors()
            if nbr.GetAtomicNum() != 1 and nbr.GetIdx() != dummy.GetIdx()
        )

        # Get default valence for anchor atom
        allowed_valence = Chem.GetPeriodicTable().GetDefaultValence(anchor.GetAtomicNum())

        if heavy_bond_count >= allowed_valence:
            raise ValueError(f'Anchor atom {anchor.GetSymbol()} fully saturated (ignoring H)')

    print('valid')

except Exception as e:
    print(f'error: {e}')
")


if [[ "$is_fragment_valid" != valid* ]]; then
    echo "     ❌ Skipping fragment — invalid or anchors not usable"
    echo "     ⚠️  Reason: $is_fragment_valid"
    echo ""
    continue
fi


    # If valid, loop over all linkers
    while read -r linker_smi; do
        echo "   🔗 Trying linker: $linker_smi"

        final_smi=$(conda run -n dl --no-capture-output \
          python ~/DiffLinker/join_linker.py \
            --fragment "$fragment_smi" \
            --linker "$linker_smi")

        if [[ "$final_smi" == ❌* || -z "$final_smi" ]]; then
            echo "     ❌ Join failed or invalid output"
            continue
        fi

        # Validate output
        is_valid=$(conda run -n dl --no-capture-output python - <<EOF
from rdkit import Chem
try:
    mol = Chem.MolFromSmiles(final_smi,sanitize=False)
    if mol:
        Chem.SanitizeMol(mol)
        print('valid')
except:
    pass
EOF
)

        if [[ "$is_valid" == "valid" ]]; then
            echo "     ✅ Joined: $final_smi"
            echo "$final_smi" >> intermediates/final_macrocycles.smi
        else
            echo "     ❌ Invalid macrocycle (valency or structure problem)"
        fi
    done <<< "$linker_smi_list"

    echo ""
done < intermediates/acyclic_with_anchors.smi

echo "✅ All valid macrocycles saved to intermediates/final_macrocycles.smi"
echo ""
echo "============================================"
echo "🎯  Pipeline Finished Successfully!"
echo "============================================"
