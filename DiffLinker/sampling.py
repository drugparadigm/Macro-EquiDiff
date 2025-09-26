import subprocess

# Input/output files
smiles_file = "/home/macrocycles/MacTransformer/datasets/final_test_dataset.txt"


output_file = "op.txt"
linkers_file = "linkers.txt"

# Clear op.txt before writing
open(output_file, "w").close()

# Step 1: Process each SMILES with generate.py
i = 0
with open(smiles_file, "r") as sf:
    for line in sf:
        smile = line.strip()
        if not smile:
            continue
        print(f"Processing: {smile}")
        
        # Call generate.py with SMILES as argument
        result = subprocess.run(
            ["python", "generate.py", smile],
            capture_output=True,
            text=True
        )
        
        # Append result to op.txt
        with open(output_file, "a") as of:
            of.write(result.stdout.strip() + "\n")
        i += 1
        if i == 5:  # stop after first line
            break

# Step 2: Run extract_linker.py
print("Running extract_linker.py...")
subprocess.run([
    "python", "extract_linker.py",
    "--macro", output_file,         # This is your generated macrocycles
    "--output", linkers_file        # Where to save linkers
])
print("Done! Extracted linkers saved to", linkers_file)

