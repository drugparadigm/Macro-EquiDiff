#!/usr/bin/env python3
import argparse
import torch
from train import MacFormer, regex_tokenize, pad_to_fixed
from rdkit import Chem

def generate_beam_search(model, src_tensor, tgt_vocab, beam_width=1, max_len=512, device='cpu'):
    model.eval()
    inv_vocab = {v: k for k, v in tgt_vocab.items()}
    sos_idx = tgt_vocab['<s>']
    eos_idx = tgt_vocab['</s>']

    beam = [([sos_idx], 0.0)]
    completed_sequences = []

    with torch.no_grad():
        for _ in range(max_len):
            new_beam = []
            for tokens, score in beam:
                if tokens[-1] == eos_idx:
                    completed_sequences.append((tokens, score))
                    continue

                tgt_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
                output = model(src_tensor, tgt_tensor)
                probs = torch.log_softmax(output[0, -1], dim=-1)
                topk_probs, topk_indices = probs.topk(beam_width)

                for i in range(beam_width):
                    next_token = topk_indices[i].item()
                    next_score = score + topk_probs[i].item()
                    new_beam.append((tokens + [next_token], next_score))

            beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:beam_width]
            if len(completed_sequences) >= beam_width:
                break

        completed_sequences += [seq for seq in beam if seq[0][-1] != eos_idx]

    completed_sequences = sorted(completed_sequences, key=lambda x: x[1], reverse=True)[:beam_width]
    decoded_sequences = [
        ''.join(inv_vocab.get(idx, '?') for idx in seq[0] if idx not in (sos_idx, eos_idx))
        for seq in completed_sequences
    ]

    return decoded_sequences

def load_model_from_checkpoint(model, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"[Checkpoint] Loaded model from {path} (Epoch {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f})")
    return model

def main():
    parser = argparse.ArgumentParser(description="Run MacFormer predictions from a SMILES string")
    parser.add_argument("--vocab", required=True, help="Path to vocab.pt file")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--smiles", required=True, help="Input SMILES string")
    parser.add_argument("--output_file", required=True, help="Path to output predictions file")
    parser.add_argument("--beam_width", type=int, default=1, help="Beam search width")
    args = parser.parse_args()

    # Load vocab
    vocab_data = torch.load(args.vocab)
    src_vocab, tgt_vocab = vocab_data['src_vocab'], vocab_data['tgt_vocab']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MacFormer(len(src_vocab), len(tgt_vocab)).to(device)

    # Load checkpoint
    model = load_model_from_checkpoint(model, args.checkpoint, device=device)

    custom_smile = args.smiles.strip()
    input_mol = Chem.MolFromSmiles(custom_smile)

    if input_mol is None:
        return

    # Tokenize & encode
    tokenized = regex_tokenize(custom_smile)
    encoded = [src_vocab.get(tok, 0) for tok in tokenized]
    padded = pad_to_fixed([encoded], max_len=512)
    src_tensor = torch.tensor(padded, dtype=torch.long).to(device)

    # Generate predictions
    predictions = generate_beam_search(model, src_tensor, tgt_vocab, beam_width=args.beam_width, device=device)

    # Write predictions to file
    with open(args.output_file, "w") as f_out:
        for smi in predictions:
            f_out.write(smi + "\n")

if __name__ == "__main__":
    main()
