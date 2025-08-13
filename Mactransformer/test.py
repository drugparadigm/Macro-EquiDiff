import torch
import pandas as pd
from train import MacFormer, regex_tokenize, pad_to_fixed

def generate_autoregressive(model, src_tensor, tgt_vocab, max_len=512, device='cpu'):
    model.eval()
    inv_vocab = {v: k for k, v in tgt_vocab.items()}
    sos_idx = tgt_vocab['<s>']
    eos_idx = tgt_vocab['</s>']
    tgt_tokens = [sos_idx]
    with torch.no_grad():
        for _ in range(max_len):
            tgt_tensor = torch.tensor([tgt_tokens], dtype=torch.long).to(device)
            output = model(src_tensor, tgt_tensor)
            next_token = output[0, -1].argmax().item()
            tgt_tokens.append(next_token)
            if next_token == eos_idx:
                break
    decoded = [inv_vocab[idx] for idx in tgt_tokens[1:-1] if idx in inv_vocab]  # Skip <s> and </s>
    return ''.join(decoded)

def load_model_from_checkpoint(model, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"[Checkpoint] Loaded model from {path} (Epoch {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f})")
    return model

def main():
    # Load vocab
    vocab_data = torch.load("vocab.pt")
    src_vocab, tgt_vocab = vocab_data['src_vocab'], vocab_data['tgt_vocab']

    # Load test CSV file
    test_df = pd.read_csv("datasets/outputtest.csv")  # Adjust path if needed
    src_smiles = test_df["src"].tolist()
    tgt_smiles = test_df["tgt"].tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MacFormer(len(src_vocab), len(tgt_vocab)).to(device)
    model = load_model_from_checkpoint(model, path="checkpoints/macformer_checkpoint_epoch_576.pth", device=device)

    correct = 0
    total = len(src_smiles)

    print("\n--- Predictions ---")
    for i, (src, tgt) in enumerate(zip(src_smiles, tgt_smiles)):
        tokenized = regex_tokenize(src)
        encoded = [src_vocab.get(tok, 0) for tok in tokenized]
        padded = pad_to_fixed([encoded], max_len=512)
        src_tensor = torch.tensor(padded, dtype=torch.long).to(device)

        pred = generate_autoregressive(model, src_tensor, tgt_vocab, device=device)

        if pred.strip() == tgt.strip():
            correct += 1

        if i < 5:  # Print a few examples
            print(f"\nExample {i+1}")
            print(f"Input SMILES : {src}")
            print(f"Target       : {tgt}")
            print(f"Prediction   : {pred}")

    acc = correct / total * 100
    print(f"\nTest Accuracy: {acc:.2f}% on {total} samples")

if __name__ == "__main__":
    main()
