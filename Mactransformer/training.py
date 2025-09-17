import re
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from keras.preprocessing.sequence import pad_sequences


def regex_tokenize(smiles: str, idx: int):
    print(f"id={idx}, type={type(smiles)} #################")
    smiles = re.sub(r"\s+", "", str(smiles)) 
    tokens = []
    i = 0
    while i < len(smiles):
        if smiles[i] == '[':
            match = re.match(r"\[[^\[\]]{1,10}\]", smiles[i:])
            if match:
                tokens.append(match.group(0))
                i += len(match.group(0))
                continue
        tokens.append(smiles[i]); i += 1
    return ['<s>'] + tokens + ['</s>']

def build_vocab(tokenized_seqs):
    vocab = {'<pad>': 0, '<s>': 1, '</s>': 2}
    idx = len(vocab)
    for seq in tokenized_seqs:
        for token in seq:
            if token not in vocab:
                vocab[token] = idx; idx += 1
    return vocab

def encode_sequences(tokenized_seqs, vocab):
    return [[vocab[token] for token in seq] for seq in tokenized_seqs]

def pad_to_fixed(seqs, max_len):
    return pad_sequences(seqs, maxlen=max_len, padding='post', value=0)

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df['src'].tolist(), df['tgt'].tolist()

def preprocess_data(filepath, max_length=512):
    src_list, tgt_list = load_data(filepath)
    src_tok = [regex_tokenize(s, i) for i, s in enumerate(src_list)]
    tgt_tok = [regex_tokenize(s, j) for j, s in enumerate(tgt_list)]
    src_vocab = build_vocab(src_tok)
    tgt_vocab = build_vocab(tgt_tok)
    src_enc = encode_sequences(src_tok, src_vocab)
    tgt_enc = encode_sequences(tgt_tok, tgt_vocab)
    src_pad, tgt_pad = pad_to_fixed(src_enc, max_length), pad_to_fixed(tgt_enc, max_length)
    return src_pad, tgt_pad, src_vocab, tgt_vocab

class CustomDataset(Dataset):
    def __init__(self, src_data, tgt_data, src_vocab, tgt_vocab):
        self.src_data, self.tgt_data = src_data, tgt_data
        self.src_vocab, self.tgt_vocab = src_vocab, tgt_vocab
        self.pad_idx = 0
    def __len__(self): return len(self.src_data)
    def __getitem__(self, idx):
        return torch.tensor(self.src_data[idx], dtype=torch.long), torch.tensor(self.tgt_data[idx], dtype=torch.long)

class MacFormer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, nhead=4,
                 num_encoder_layers=4, num_decoder_layers=4, dim_feedforward=512,
                 dropout=0.1, max_len=5000):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.register_buffer("positional_encoding", self.get_sinusoidal_encoding(max_len, d_model))
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.ones(sz, sz), diagonal=1).bool()
    def forward(self, src, tgt):
        src_emb = self.src_embedding(src) + self.positional_encoding[:, :src.size(1), :]
        tgt_emb = self.tgt_embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :]
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        out = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
        return self.fc_out(out)
    def get_sinusoidal_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # shape: (1, max_len, d_model)
def train_model(model, dataloader, criterion, optimizer, src_vocab, tgt_vocab, epochs=20, device='cpu'):
    model.train()
    a = -1
    for epoch in range(epochs):
        total_loss = 0
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            # Inside train_model()
            outputs = model(src, tgt[:, :-1])  # input
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), tgt[:, 1:].reshape(-1))  # target
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
        save_checkpoint(
            model, optimizer, epoch + 1, avg_loss,
            path=f"pep_checkpoints/macformer_checkpoint_epoch_{epoch+1}.pth"
        )

def evaluate_model(model, dataloader, criterion, device, pad_idx):
    model.eval()
    total_loss = total_correct = total_tokens = 0
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            outputs = model(src, tgt)
            loss = criterion(outputs.view(-1, outputs.size(-1)), tgt.view(-1))
            total_loss += loss.item()
            pred = outputs.argmax(dim=-1)
            mask = (tgt != pad_idx)
            total_correct += ((pred == tgt) & mask).sum().item()
            total_tokens += mask.sum().item()
    return total_loss/len(dataloader), (total_correct/total_tokens if total_tokens else 0)

def predict_smile(model, dataset, src_vocab, tgt_vocab, index, device):
    model.eval()
    inv_tgt = {v: k for k, v in tgt_vocab.items()}
    with torch.no_grad():
        src, tgt = dataset[index]
        out = model(src.unsqueeze(0).to(device), tgt.unsqueeze(0).to(device))
        pred_ids = out.argmax(dim=-1)[0].cpu().tolist()

    src_str = ''.join(inv_tgt[idx] for idx in src.tolist() if idx != dataset.pad_idx)
    tgt_str = ''.join(inv_tgt[idx] for idx in tgt.tolist() if idx != dataset.pad_idx)
    pred_str = []
    for idx in pred_ids:
        if idx == tgt_vocab.get('</s>'):
            pred_str.append('</s>'); break
        if idx != dataset.pad_idx:
            pred_str.append(inv_tgt.get(idx, '?'))
    return src_str, tgt_str, ''.join(pred_str)

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

def save_checkpoint(model, optimizer, epoch, loss, path="pep_checkpoints/macformer_checkpoint_epoch_{epoch}.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, path)
    print(f"[Checkpoint] Saved at epoch {epoch} to {path}")

def load_checkpoint(path, model, optimizer, device='cpu'):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"[Checkpoint] Loaded from {path} (epoch {start_epoch}, loss {loss:.4f})")
    return model, optimizer, start_epoch, loss

def main():
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    train_fp = 'datasets/pept_data.csv'
    src_data, tgt_data, src_vocab, tgt_vocab = preprocess_data(train_fp, max_length=512)
    # Save vocabularies to vocab.pt
    torch.save({'src_vocab': src_vocab, 'tgt_vocab': tgt_vocab}, 'vocab_pep.pt')
    # print("[Vocab] Saved vocabularies to vocab.pt")
    ds = CustomDataset(src_data, tgt_data, src_vocab, tgt_vocab)
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MacFormer(len(src_vocab), len(tgt_vocab)).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=ds.pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    checkpoint_path = ""
    # --- 🔹 Resume from epoch 18 checkpoint ---
    if os.path.exists(checkpoint_path):
        model, optimizer, start_epoch, prev_loss = load_checkpoint(checkpoint_path, model, optimizer, device=device)
        print(f"Resuming training from epoch {start_epoch+1} ...")
    else:
        print("[Checkpoint] No checkpoint found, starting from scratch.")
        start_epoch = 0

    # Continue training from epoch 19 up to total 1000
    remaining_epochs = 40 - start_epoch
    train_model(model, dl, criterion, optimizer, src_vocab, tgt_vocab, epochs=remaining_epochs, device=device)

    # test_fp = '/content/outputtest.csv'
    # ts_data, tt_data, ts_vocab, tt_vocab = preprocess_data(test_fp, max_length=512)
    # test_ds = CustomDataset(ts_data, tt_data, ts_vocab, tt_vocab)
    # test_dl = DataLoader(test_ds, batch_size=32)
    # test_loss, test_acc = evaluate_model(model, test_dl, criterion, device, test_ds.pad_idx)
    # print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
    
    # for i in range(min(3, len(test_ds))):
    #     src_sm, tgt_sm, pred_sm = predict_smile(model, test_ds, ts_vocab, tt_vocab, i, device)
    #     print(f"\nExample {i+1}\nSrc: {src_sm}\nTgt: {tgt_sm}\nPred: {pred_sm}")
    # --- Predict on custom SMILES string ---
    # custom_smile = "C1=CC(=CC=C1)NC2=C(C=NC(=N2)NC3=CC=C(C=C3)OCCN4CCCC4)C"
    # tokenized = regex_tokenize(custom_smile)
    # encoded = [src_vocab.get(tok, 0) for tok in tokenized]
    # padded = pad_to_fixed([encoded], max_len=512)

    # src_tensor = torch.tensor(padded, dtype=torch.long).to(device)
    # predicted_smile = generate_autoregressive(model, src_tensor, tgt_vocab, max_len=512, device=device)

    # print(f"\nPrediction on SMILES '{custom_smile}':")
    # print(f"Tokenized Input : {tokenized}")
    # print(f"Predicted Output: {predicted_smile}")


if __name__ == "__main__":
    main()