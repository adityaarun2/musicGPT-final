import torch
from torch.utils.data import Dataset
from collections import Counter
import json

class MIDITokenDataset(Dataset):
    def __init__(self, filepath, token_to_id, max_length=1024):
        self.samples = []
        self.max_length = max_length
        self.token_to_id = token_to_id

        with open(filepath, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                ids = [token_to_id.get(tok, token_to_id["<unk>"]) for tok in tokens]

                # Skip too short lines
                if len(ids) < 10:
                    continue

                # Chunk long sequences
                for i in range(0, len(ids), max_length):
                    chunk = ids[i:i + max_length]
                    if len(chunk) > 1:
                        self.samples.append(chunk)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx][:-1]
        y = self.samples[idx][1:]
        return {
            "input_ids": torch.tensor(x, dtype=torch.long),
            "labels": torch.tensor(y, dtype=torch.long)
        }

def build_vocab(filepath, vocab_path="vocab.json"):
    vocab = Counter()
    with open(filepath) as f:
        for line in f:
            vocab.update(line.strip().split())

    vocab_list = ["<pad>", "<unk>"] + sorted(vocab.keys())
    token_to_id = {tok: i for i, tok in enumerate(vocab_list)}
    id_to_token = {i: tok for tok, i in token_to_id.items()}

    with open(vocab_path, "w") as f:
        json.dump(token_to_id, f, indent=2)

    return token_to_id, id_to_token

def pad_collate(batch):
    input_seqs = [item["input_ids"] for item in batch]
    label_seqs = [item["labels"] for item in batch]
    
    max_len = max([len(seq) for seq in input_seqs])
    
    padded_inputs = torch.stack([torch.cat([seq, torch.full((max_len - len(seq),), 0)]) for seq in input_seqs])
    padded_labels = torch.stack([torch.cat([seq, torch.full((max_len - len(seq),), 0)]) for seq in label_seqs])
    
    return {"input_ids": padded_inputs.long(), "labels": padded_labels.long()}


if __name__ == "__main__":
    token_to_id, id_to_token = build_vocab("all_tokenized_miditok.txt")
    dataset = MIDITokenDataset("all_tokenized_miditok.txt", token_to_id)
    print("Number of samples:", len(dataset))
    print("Sample 0:", dataset[0])