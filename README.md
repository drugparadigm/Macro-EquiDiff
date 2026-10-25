# Macro-EquiDiff (MED)

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Conda](https://img.shields.io/badge/Conda-Required-green)](https://docs.conda.io/)
[![GPU Recommended](https://img.shields.io/badge/GPU-Recommended-orange)](https://www.nvidia.com/)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

Macrocyclic compounds are important in drug discovery as they can modulate protein–protein interactions and enhance selectivity. Their structural complexity enables access to chemical space beyond traditional small molecules, but designing feasible macrocycles remains difficult. <br>
Current computational methods often fail to generate macrocycles with proper drug-like properties. Here, we present Macro-Equi-Diff (MED), a deep learning framework combining transformer-based site identification, an E(3)-equivariant diffusion model (EDM) for linker creation, a fragment–linker attachment and pharmacokinetic validation module. MED transforms acyclic molecules into structurally consistent macrocycles. MED was tested on the ZINC dataset, achieving high validity (93.82%), uniqueness (99.94%), macrocyclization (99.92%), and linker novelty (82.81%). MED improves upon previous methods that lack macrocyclic geometry context. <br>
As a case study, MED was used to macrocyclize four acyclic drugs targeting the JAK2 protein. The resulting macrocycles showed favourable predicted docking scores, emphasizing MED as a reliable method for expanding the macrocyclic chemical space.

**Release:** this README describes the code version that matches the manuscript results. See the repository's *Releases* page for the tagged version.

---

## Contents

1. [Overview of the pipeline](#overview-of-the-pipeline)
2. [Repository layout](#repository-layout)
3. [Installation](#installation)
4. [Setup: checkpoints and datasets](#setup-checkpoints-and-datasets)
5. [Usage](#usage)
6. [Output files](#output-files)
7. [Minimal end-to-end example](#minimal-end-to-end-example)
8. [Computing the benchmark metrics](#computing-the-benchmark-metrics)
9. [Reproducibility notes](#reproducibility-notes)
10. [Citation and acknowledgements](#citation-and-acknowledgements)
11. [License](#license)
12. [Contact](#contact)

---

## Overview of the pipeline

Input: one acyclic molecule as a SMILES string. `MED_inference.py` runs these steps for each input:

1. **3D structure.** The SMILES is converted to a 3D structure with RDKit (hydrogens added, one embedding) and written to `user_input.sdf`.
2. **Attachment-site prediction (MacTransformer).** The transformer predicts the two cyclization sites, returned as a SMILES with two `*` marks. Five randomized SMILES of the input are used (`N_AUGMENT = 5`) and decoding uses `--beam_width 1`.
3. **Linker generation (EDM).** The equivariant diffusion model generates linkers between the two fragments in 3D. It uses the pretrained DiffLinker GEOM checkpoint and the linker-size model. Nothing was retrained for MED.
4. **Linker extraction.** The linker is read from each generated SDF. The first *n* heavy atoms of the generated molecule are the acyclic fragment (*n* = heavy atoms of the input). The remaining atoms are the linker. The two linker atoms bonded to the fragment are the anchor points, and the farthest-apart pair is used if more than two are found.
5. **Fragment–linker attachment.** Each `*` on the fragment is joined to the matching `*` on the linker. Only products with a ring of 11 or more atoms are kept as macrocycles.
6. **Property filtering and ranking.** Physicochemical properties, simple threshold-based ADMET flags, a PAINS alert, QED and a drug-likeness score are computed. The top five macrocycles per input are written to a CSV.

If a run produces no valid macrocycle, steps 3–6 are retried up to 5 times.

---

## Repository layout

```
Macro-EquiDiff/
├── MED_inference.py              # full pipeline (entry point)
├── pipeline_metrics.py           # property / ADMET table for a few SMILES
├── inference_test.txt            # example input (one SMILES per line)
├── MacTransformer.yml            # conda environment "mact"
├── EDM.yml                       # conda environment "edm"
├── MacTransformer/
│   ├── pipeline_predict.py       # transformer inference
│   ├── vocab.pt                  # vocabulary (included)
│   ├── macTransformer_checkpoint.pth   # <-- PUT THE TRANSFORMER CHECKPOINT HERE
│   └── datasets/data/            # train.csv, data.csv, zinc_test.txt
└── EDM/
    ├── generate.py               # diffusion-based linker generation
    ├── compute_metrics.py        # benchmark metrics (validity, uniqueness, novelty, ...)
    ├── configs/geom_EDM.yml
    └── models/                   # <-- CREATE THIS FOLDER AND PUT THE EDM CHECKPOINTS HERE
        ├── geom_EDM.ckpt
        └── geom_size_gnn.ckpt
```

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/drugparadigm/Macro-EquiDiff.git
   cd Macro-EquiDiff
   ```

2. Create the two conda environments:
   ```bash
   conda env create -f MacTransformer.yml   # creates the environment "mact"
   conda env create -f EDM.yml              # creates the environment "edm"
   ```

`MED_inference.py` calls the two environments by name (`conda run -n mact ...` and `conda run -n edm ...`), so both must exist with exactly these names.

---

## Setup: checkpoints and datasets

### Model checkpoints

| Model | Used for | Download | Save as |
|---|---|---|---|
| Diffusion model (DiffLinker, GEOM) | Linker generation | [geom_difflinker.ckpt](https://zenodo.org/record/7121300/files/geom_difflinker.ckpt?download=1) | `EDM/models/geom_EDM.ckpt` (rename after download) |
| Linker-size model | Predicts the linker size | [geom_size_gnn.ckpt](https://zenodo.org/record/7121300/files/geom_size_gnn.ckpt?download=1) | `EDM/models/geom_size_gnn.ckpt` |
| MacTransformer | Attachment-site prediction | *add the link to your checkpoint file here* | `MacTransformer/macTransformer_checkpoint.pth` |

Notes:
- The two diffusion checkpoints are the pretrained DiffLinker models, used as released. DiffLinker's model archive is also at <https://doi.org/10.5281/zenodo.7775568>.
- `EDM/models/` is not part of the repository. Create it before downloading.
- The file names above are the ones `MED_inference.py` expects. If you save a file under a different name, change the path in `run_EDM` or `run_macformer_on_smiles_in_memory`.
- `MacTransformer/vocab.pt` is already included.

### Datasets

| Dataset | Used for | Link |
|---|---|---|
| GEOM fragment–linker pairs (from DiffLinker) | Training data of the pretrained diffusion model | <https://zenodo.org/records/7121278> |
| ZINC test set (from Macformer) | Benchmark test set (5,551 pairs) | <https://github.com/yydiao1025/Macformer/tree/main/data/ZINC> |
| Transformer training data | Training and test files used for MacTransformer | `MacTransformer/datasets/data/` in this repository |

### Related code

- DiffLinker: <https://github.com/igashov/DiffLinker>
- Macformer: <https://github.com/yydiao1025/Macformer>

---

## Usage

1. **Activate an environment that has RDKit, pandas and PyTorch** (the pipeline script itself runs there; it launches the two model environments on its own):
   ```bash
   conda activate mact
   ```

2. **Prepare the input SMILES.** Put one SMILES per line in `inference_test.txt`, or pass your own file:
   ```bash
   python MED_inference.py --input_file inference_test.txt
   ```

3. **Run the pipeline** (the command above). The default input file is `inference_test.txt`.

---

## Output files

Everything is written to the `intermediates/` folder. For input number *i* there is a sub-folder `intermediates/input_<i>/`:

| File | Content |
|---|---|
| `user_input.sdf` | 3D structure of the input molecule |
| `valid_macformer_smiles.txt` | Transformer outputs (all five randomized inputs) |
| `cyclizer_smiles.txt` | Valid fragment SMILES with two `*` attachment points |
| `diff_linker_ops/*.sdf` | Raw diffusion outputs |
| `linkers.sdf` | Linkers extracted from the diffusion outputs |
| `extracted_linkers.txt` | Linker SMILES with two `*` |
| `ops.txt` | Macrocycle SMILES after fragment–linker attachment |
| `final_macros.txt` | Macrocycles with a ring of 11 or more atoms |
| `top_5_macrors.csv` | Properties of the input and the top five macrocycles |

The top macrocycle of each input is collected in `intermediates/top_performing_per_input.csv`.

---

## Minimal end-to-end example

```bash
conda activate mact
echo "<acyclic SMILES>" > inference_test.txt      # or use the provided example
python MED_inference.py --input_file inference_test.txt
```

Expected result: the folder `intermediates/input_0/` with the files listed above, and `intermediates/top_performing_per_input.csv` with one row per input that gave a macrocycle. Because diffusion sampling is stochastic, the exact macrocycles differ from run to run.

---

## Computing the benchmark metrics

`EDM/compute_metrics.py` computes validity, uniqueness, novelty and the other metrics on a file of generated molecules. Run it from the `EDM` folder inside the `edm` environment:

```bash
python compute_metrics.py <data_set> <generated_smiles_file> <training_linkers_file> <n_cores> <verbose> <restrict> <pains_smarts_file> <method>
```

| Argument | Meaning |
|---|---|
| `data_set` | `ZINC` (the benchmark used here), or `CASF`, `GEOM`, `MOAD` |
| `generated_smiles_file` | Space-separated lines: fragments, original molecule, generated molecule, generated linker |
| `training_linkers_file` | Training-set linkers, one SMILES per line |
| `n_cores` | Number of CPU cores |
| `verbose` | Print the settings (any non-empty value is treated as true) |
| `restrict` | Use only the first N lines, or `None` for all |
| `pains_smarts_file` | File with the PAINS SMARTS |
| `method` | `diffusion` |

Definitions used in the manuscript:
- **Validity:** percentage of generated molecules that parse, sanitize and contain the input fragments.
- **Uniqueness:** percentage of unique molecules among the generated valid molecules, per input.
- **Novelty of the linker:** percentage of linkers not in the training set, among the generated valid molecules. Linkers are compared after stereochemistry removal and tautomer canonicalization, applied to both the generated and the training linkers.

The results are also written to `<generated_smiles_file>_summary.csv`.

---

## Reproducibility notes

- No random seed is set. Diffusion sampling and the randomized SMILES inputs differ from run to run, so repeated runs give slightly different numbers.
- The transformer is decoded with a beam width of 1.
- GPU use: the pipeline selects GPUs through `CUDA_VISIBLE_DEVICES`. Edit this line in `MED_inference.py` to match your machine.

---

## Citation and acknowledgements

If you use this pipeline in your research, please cite the MED manuscript (bioRxiv 2026.02.05.703948; add the final reference when available) and the works MED builds on:

- Igashov, I. *et al.* Equivariant 3D-conditional diffusion model for molecular linker design. *Nat. Mach. Intell.* **6**, 417–427 (2024). doi:10.1038/s42256-024-00815-9
- Diao, Y. *et al.* Macformer. *Nat. Commun.* (2023). doi:10.1038/s41467-023-40219-8

The diffusion model code and checkpoints come from DiffLinker, and the ZINC test set and baseline results come from Macformer. Please respect their licenses.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

For questions or support, open an issue on the repository or contact the maintainers at `<your-contact-email>`.
