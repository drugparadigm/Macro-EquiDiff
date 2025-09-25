# Macro-EquiDiff (MED)

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Conda](https://img.shields.io/badge/Conda-Required-green)](https://docs.conda.io/)
[![GPU Recommended](https://img.shields.io/badge/GPU-Recommended-orange)](https://www.nvidia.com/)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

Macrocyclic compounds are important in drug discovery as they can modulate protein–protein interactions and enhance selectivity. Their structural complexity enables access to chemical space beyond traditional small molecules, but designing feasible macrocycles remains difficult. <br>
Current computational methods often fail to generate macrocycles with proper drug-like properties. Here, we present Macro-Equi-Diff (MED), a deep learning framework combining transformer-based site identification, an E(3)-equivariant diffusion model (EDM) for linker creation, a fragment–linker attachment and pharmacokinetic validation module. MED transforms acyclic molecules into structurally consistent macrocycles. MED was tested on the ZINC dataset, achieving high validity (93.82%), uniqueness (99.94%), macrocyclization (99.94%), and linker novelty (65.66%). MED improves upon previous methods that lack macrocyclic geometry context. <br>
As a case study, MED was used to macrocyclize four acyclic drugs targeting the JAK2 protein. The resulting macrocycles demonstrated strong binding affinities, emphasizing MED as a reliable method for expanding the macrocyclic chemical space.


---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/drugparadigm/Macro-EquiDiff.git
   cd Macro-EquiDiff
   ```

2. Create the required environments:
   ```bash
   conda env create -f MacTransformer.yml
   conda env create -f EDM.yml
   ```

---

## Usage

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

## Output

An **intermediates** folder is created where the intermediates and the output macrocycles are stored



## Notes

- Ensure you are inside the correct environment (`macformer_env`) before running inference.
- The `EDM` environment is required for diffusion-based model components.
- For best performance, use a GPU-enabled system.

---

## Citation

If you use this pipeline in your research, please cite:
> [Add your paper/preprint/patent reference here if applicable]

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
