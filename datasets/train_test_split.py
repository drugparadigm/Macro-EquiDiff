import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def split(table_path, train_path, val_path, test_path):
    table = pd.read_csv(table_path)
    table = table.drop_duplicates(['molecule', 'linker'])

    # First split: Train (80%) + Temp (20%)
    train_data, temp_data = train_test_split(
        table,
        test_size=0.2,
        random_state=42
    )

    # Second split: Temp -> Val (10%) + Test (10%)
    val_data, test_data = train_test_split(
        temp_data,
        test_size=0.5,
        random_state=42
    )

    print(f"Train size: {len(train_data)}")
    print(f"Val size: {len(val_data)}")
    print(f"Test size: {len(test_data)}")

    train_data.to_csv(train_path, index=False)
    val_data.to_csv(val_path, index=False)
    test_data.to_csv(test_path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generated-splits', type=str, required=True)
    parser.add_argument('--train', type=str, required=True)
    parser.add_argument('--val', type=str, required=True)
    parser.add_argument('--test', type=str, required=True)
    args = parser.parse_args()

    split(
        table_path=args.generated_splits,
        train_path=args.train,
        val_path=args.val,
        test_path=args.test,
    )
