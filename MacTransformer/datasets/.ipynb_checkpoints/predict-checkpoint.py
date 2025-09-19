import torch
from train import MacFormer, regex_tokenize, pad_to_fixed
from rdkit import Chem

def greedy_predict(model, src_tensor, tgt_vocab, max_len=512, device='cpu'):
    model.eval()
    inv_vocab = {v: k for k, v in tgt_vocab.items()}
    sos_idx = tgt_vocab['<s>']
    eos_idx = tgt_vocab['</s>']

    tokens = [sos_idx]
    with torch.no_grad():
        for _ in range(max_len):
            tgt_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
            output = model(src_tensor, tgt_tensor)
            next_token = torch.argmax(output[0, -1]).item()
            tokens.append(next_token)
            if next_token == eos_idx:
                break

    return ''.join(inv_vocab.get(idx, '?') for idx in tokens if idx not in (sos_idx, eos_idx))


def clean_prediction(smiles):
    return smiles.replace("(*)", "").replace('[*]', '').replace('*', '')


def load_model_from_checkpoint(model, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"[Checkpoint] Loaded model from {path} (Epoch {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f})")
    return model


def main():
    input_file = "/home/macrocycles/datasets/final_test_dataset.txt"
    output_file = "/home/macrocycles/datasets/predicted_smiles.txt"

    # Load vocab
    vocab_data = torch.load("vocab.pt")
    src_vocab, tgt_vocab = vocab_data['src_vocab'], vocab_data['tgt_vocab']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MacFormer(len(src_vocab), len(tgt_vocab)).to(device)

    # Load checkpoint
    model = load_model_from_checkpoint(model, path="macformer_checkpoint_epoch_18.pth", device=device)

    with open(input_file, "r") as f:
        smiles_list = [line.strip() for line in f if line.strip()]

    predictions = []
    for smi in smiles_list:
        tokenized = regex_tokenize(smi)
        encoded = [src_vocab.get(tok, 0) for tok in tokenized]
        padded = pad_to_fixed([encoded], max_len=512)
        src_tensor = torch.tensor(padded, dtype=torch.long).to(device)

        pred = greedy_predict(model, src_tensor, tgt_vocab, device=device)
        cleaned_pred = clean_prediction(pred)

        # Validate SMILES
        if Chem.MolFromSmiles(cleaned_pred):
            predictions.append(cleaned_pred)
        else:
            predictions.append("INVALID")

    with open(output_file, "w") as f:
        for pred in predictions:
            f.write(pred + "\n")

    print(f"✅ Predictions saved to {output_file}")


if __name__ == "__main__":
    main()
