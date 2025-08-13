# import torch
# from train import MacFormer, regex_tokenize, pad_to_fixed
# from rdkit import Chem
# from rdkit.Chem import MolToSmiles
# import os

# def generate_beam_search(model, src_tensor, tgt_vocab, beam_width=1, max_len=512, device='cpu'):
#     model.eval()
#     inv_vocab = {v: k for k, v in tgt_vocab.items()}
#     sos_idx = tgt_vocab['<s>']
#     eos_idx = tgt_vocab['</s>']

#     beam = [([sos_idx], 0.0)]
#     completed_sequences = []

#     with torch.no_grad():
#         for _ in range(max_len):
#             new_beam = []
#             for tokens, score in beam:
#                 if tokens[-1] == eos_idx:
#                     completed_sequences.append((tokens, score))
#                     continue

#                 tgt_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
#                 output = model(src_tensor, tgt_tensor)
#                 probs = torch.log_softmax(output[0, -1], dim=-1)
#                 topk_probs, topk_indices = probs.topk(beam_width)

#                 for i in range(beam_width):
#                     next_token = topk_indices[i].item()
#                     next_score = score + topk_probs[i].item()
#                     new_beam.append((tokens + [next_token], next_score))

#             beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:beam_width]
#             if len(completed_sequences) >= beam_width:
#                 break

#         completed_sequences += [seq for seq in beam if seq[0][-1] != eos_idx]

#     completed_sequences = sorted(completed_sequences, key=lambda x: x[1], reverse=True)[:beam_width]
#     decoded_sequences = [
#         ''.join(inv_vocab.get(idx, '?') for idx in seq[0] if idx not in (sos_idx, eos_idx))
#         for seq in completed_sequences
#     ]

#     return decoded_sequences


# def load_model_from_checkpoint(model, path, device):
#     checkpoint = torch.load(path, map_location=device)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     print(f"[Checkpoint] Loaded model from {path} (Epoch {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f})")
#     return model


# def is_same_molecule(smiles1, smiles2):
#     mol1 = Chem.MolFromSmiles(smiles1)
#     mol2 = Chem.MolFromSmiles(smiles2)
#     if mol1 is None or mol2 is None:
#         return False
#     return Chem.MolToSmiles(mol1, canonical=True) == Chem.MolToSmiles(mol2, canonical=True)


# def clean_prediction(smiles):
#     return smiles.replace("()","").replace('[]', '').replace('*', '')


# def main():
#     input_file = "/home/macrocycles/MacTransformer/datasets/final_test_dataset.txt"
#     output_file = "/home/macrocycles/MacTransformer/datasets/predictions.txt"

#     # Load vocab
#     vocab_data = torch.load("vocab.pt")
#     src_vocab, tgt_vocab = vocab_data['src_vocab'], vocab_data['tgt_vocab']

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = MacFormer(len(src_vocab), len(tgt_vocab)).to(device)

#     # Load checkpoint
#     model = load_model_from_checkpoint(model, path="checkpoints/macformer_checkpoint_epoch_21.pth", device=device)

#     # Custom SMILES input
#     with open(input_file,"r") as f:
#         smiles_list=[l for l in f]
#     with open(output_file, "w") as f_out:
#         for idx, custom_smile in enumerate(smiles_list, start=1):
#             print(f"\n[{idx}/{len(smiles_list)}] Processing: {custom_smile}")
#             input_mol = Chem.MolFromSmiles(custom_smile)
    
#             if input_mol is None:
#                 print(f"❌ Invalid input SMILES: {custom_smile}")
#                 continue
    
#             # Tokenize & encode
#             tokenized = regex_tokenize(custom_smile)
#             encoded = [src_vocab.get(tok, 0) for tok in tokenized]
#             padded = pad_to_fixed([encoded], max_len=512)
#             src_tensor = torch.tensor(padded, dtype=torch.long).to(device)
    
#             # Generate predictions
#             predictions = generate_beam_search(model, src_tensor, tgt_vocab, beam_width=1, device=device)
    
#             # Write predictions to file
#             for smi in predictions:
#                 f_out.write(smi + "\n")
    



    
#     # custom_smile = "CCC=CC1CC1(NC(=O)C(NC(=O)CNC(=O)OC(C)(C)C)c1ccc(Oc2cc(-c3ccccc3)nc3cc(OC)ccc23)cc1)C(=O)[O-]"
#     # tokenized = regex_tokenize(custom_smile)
#     # encoded = [src_vocab.get(tok, 0) for tok in tokenized]
#     # padded = pad_to_fixed([encoded], max_len=512)
#     # src_tensor = torch.tensor(padded, dtype=torch.long).to(device)

#     # # Generate predictions
#     # predictions = generate_beam_search(model, src_tensor, tgt_vocab, beam_width=1, device=device)

#     # print(f"\nInput SMILES: {custom_smile}")
#     # input_mol = Chem.MolFromSmiles(custom_smile)

#     # # Save only if valid outputs exist
#     # with open(smi_filename, "w") as f:
#     #     for smi in predictions:
#     #         f.write(smi + "\n")

#     # print(f"\n✅ Written {len(predictions)} valid matching SMILES to: {smi_filename}")


# if __name__ == "__main__":
#     main()


# predict.py

import torch
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
    decoded = [inv_vocab[idx] for idx in tgt_tokens]
    return ''.join(decoded)

def load_model_from_checkpoint(model, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"[Checkpoint] Loaded model from {path} (Epoch {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f})")
    return model

def main(smiles_list):
    if not smiles_list:
        print("No SMILES provided.")
        return []

    # Load vocab
    vocab_data = torch.load("/home/macrocycles/MacTransformer/vocab.pt")
    src_vocab, tgt_vocab = vocab_data['src_vocab'], vocab_data['tgt_vocab']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MacFormer(len(src_vocab), len(tgt_vocab)).to(device)

    # Load checkpoint
    model = load_model_from_checkpoint(model, path="/home/macrocycles/MacTransformer/macformer_checkpoint_epoch_18.pth", device=device)

    output_file = "/home/macrocycles/drug_sinter/valid_macformer_smiles.txt"
    predictions = []

    for custom_smile in smiles_list:
        tokenized = regex_tokenize(custom_smile)
        encoded = [src_vocab.get(tok, 0) for tok in tokenized]
        padded = pad_to_fixed([encoded], max_len=512)
        src_tensor = torch.tensor(padded, dtype=torch.long).to(device)

        predicted = generate_autoregressive(model, src_tensor, tgt_vocab, device=device)

        print(f"\nInput SMILES   : {custom_smile}")
        print(f"Predicted Output: {predicted}")

        predictions.append(predicted)

        with open(output_file, "a") as f:
            f.write(f"{predicted}\n")

    print(f"\nPredictions saved to {output_file}")
    return predictions

if __name__ == "__main__":
    # sys.argv[0] is the script name, so skip it
    smiles_list = sys.argv[1:]
    main(smiles_list)