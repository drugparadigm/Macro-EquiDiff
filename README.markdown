# Macro-EquiDiff (MED) Pipeline

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Conda](https://img.shields.io/badge/Conda-Required-green)](https://docs.conda.io/)
[![GPU Recommended](https://img.shields.io/badge/GPU-Recommended-orange)](https://www.nvidia.com/)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

Macro-EquiDiff (MED) is a pipeline designed for generating and evaluating macrocyclic molecules. It leverages **MacFormer** and **Equivariant Diffusion Models (EDM)** to perform structure-based macrocycle design and inference from SMILES strings.

---

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd <your-repo>
   ```

2. Create the required environments:
   ```bash
   conda env create -f MacTransformer.yml
   conda env create -f EDM.yml
   ```

---

## ▶️ Usage

1. **Activate the environment**:
   ```bash
   conda activate macformer_env
   ```

2. **Prepare input SMILES**:
   Place a list of SMILES strings in a text file named:
   ```
   inference_text.txt
   ```

3. **Run the pipeline**:
   ```bash
   python inference.py
   ```

---

## 📂 Input & Output

**Input**:
- `inference_text.txt` → a plain text file containing SMILES strings (one per line).

**Output**:
- Generated macrocyclic molecules and corresponding linkers will be saved automatically in the `results` directory in `.sdf` and `.xyz` formats.

---

## ⚡ Example

**Input** (`inference_text.txt`):
```
CCOCCN(CC)CCO
CC1=CC=CC=C1
O=C1C=CCCOCCC(=O)N
```

**Run**:
```bash
python inference.py
```

**Output**:
- Candidate macrocycles saved in `.sdf` and `.xyz` formats in the `results` directory.

---

## 💡 Notes

- Ensure you are inside the correct environment (`macformer_env`) before running inference.
- The `EDM` environment is required for diffusion-based model components.
- For best performance, use a GPU-enabled system.

---

## 📜 Citation

If you use this pipeline in your research, please cite:
> [Add your paper/preprint/patent reference here if applicable]

---

## Troubleshooting

- **Environment Issues**: Ensure Conda is up-to-date and verify compatibility of `MacTransformer.yml` and `EDM.yml` with your system.
- **Missing Input File**: Confirm `inference_text.txt` exists in the working directory and contains valid SMILES strings.
- **Script Errors**: Verify the `macformer_env` is activated and all dependencies are installed. Check error logs for details.

---

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit changes and push:
   ```bash
   git commit -m "Add your feature description"
   git push origin feature/your-feature-name
   ```
4. Open a pull request.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

For questions or support, open an issue on the repository or contact the maintainers at `<your-contact-email>`.