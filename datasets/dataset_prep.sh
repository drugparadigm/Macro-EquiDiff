#!/bin/bash
set -e  # stop if any command fails

# python fragmentation.py

# python generate_macrocycle_conformers.py

# python train_test_split.py \
#   --generated-splits "macrocycle_table.csv" \
#   --train "data/macrocycle_train_table_split.csv" \
#   --val "data/macrocycle_val_table_split.csv" \
#   --test "data/macrocycle_test_table_split.csv"

# python prepare_macrocycles_dataset.py \
#   --table macrocycle_table.csv \
#   --sdf macrocycle_conformers.sdf \
#   --out-mol-sdf macrocycle_mol.sdf \
#   --out-frag-sdf macrocycle_frag.sdf \
#   --out-link-sdf macrocycle_link.sdf \
#   --out-table macrocycle_out_table.csv

# python prepare_macrocycles_dataset.py \
#   --table data/macrocycle_train_table_split.csv \
#   --sdf macrocycle_conformers.sdf \
#   --out-mol-sdf macrocycle_train_mol.sdf \
#   --out-frag-sdf macrocycle_train_frag.sdf \
#   --out-link-sdf macrocycle_train_link.sdf \
#   --out-table macrocycle_train_table.csv

python prepare_macrocycles_dataset.py \
  --table data/macrocycle_val_table_split.csv \
  --sdf macrocycle_conformers.sdf \
  --out-mol-sdf macrocycle_val_mol.sdf \
  --out-frag-sdf macrocycle_val_frag.sdf \
  --out-link-sdf macrocycle_val_link.sdf \
  --out-table macrocycle_val_table.csv


