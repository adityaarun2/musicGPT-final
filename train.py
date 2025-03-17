import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MIDITokenDataset, build_vocab
from music_transformer import MusicTransformer
from dataset import pad_collate

# === Config ===
DATA_PATH = "all_tokenized_miditok.txt"
VOCAB_PATH = "vocab.json"
MAX_SEQ_LEN = 512
BATCH_SIZE = 2
EPOCHS = 5
LR = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Step 1: Build vocab + dataset ===
print("🔤 Building vocabulary and dataset...")
token_to_id, id_to_token = build_vocab(DATA_PATH, vocab_path=VOCAB_PATH)
dataset = MIDITokenDataset(DATA_PATH, token_to_id, max_length=MAX_SEQ_LEN)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate)

# === Step 2: Initialize model ===
print("🎵 Initializing model...")
model = MusicTransformer(
    vocab_size=len(token_to_id),
    d_model=256,
    nhead=8,
    num_layers=6,
    dim_feedforward=512,
    dropout=0.1,
    max_seq_len=MAX_SEQ_LEN
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # 0 = <pad>

# === Step 3: Training loop ===
print("🚀 Starting training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for batch in pbar:
        x = batch['input_ids'].to(DEVICE)
        y = batch['labels'].to(DEVICE)

        logits = model(x)
        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=total_loss / (pbar.n + 1))

    print(f"✅ Epoch {epoch+1} finished. Avg Loss: {total_loss / len(loader):.4f}")

# === Step 4: Save model + vocab ===
torch.save(model.state_dict(), "music_transformer_genre.pth")
print("💾 Model saved to music_transformer_genre.pth")
