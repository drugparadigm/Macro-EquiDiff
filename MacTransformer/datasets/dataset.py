import pandas as pd

def process_fragments(input_csv, output_csv):
    """
    Reads a CSV file with a 'fragments' column, and creates a new CSV with:
    - 'src': fragments with '*' removed
    - 'tgt': original fragments
    """
    # Read input CSV
    df = pd.read_csv(input_csv)

    if "fragments" not in df.columns:
        raise ValueError("Input CSV must have a 'fragments' column.")

    # Create new dataframe
    new_df = pd.DataFrame({
        "src": df["fragments"].replace("*", "").replace("(*)","").replace("[*]",""),
        "tgt": df["fragments"]
    })

    # Save to CSV
    new_df.to_csv(output_csv, index=False)
    print(f" Processed CSV saved to {output_csv}")


if __name__ == "__main__":
    #  set your paths here
    input_csv = "data/macrocycle_data.csv"
    output_csv = "data/data.csv"

    process_fragments(input_csv, output_csv)
