# c:\quasarv4\quasar\utils.py

import torch

# This is a placeholder for a real tokenizer.
# In a full implementation, this would be a proper BPE or WordPiece tokenizer.
class SimpleTokenizer:
    def __init__(self, corpus):
        self.vocab = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
        self.rev_vocab = {v: k for k, v in self.vocab.items()}
        self._build_vocab(corpus)

    def _build_vocab(self, corpus):
        idx = len(self.vocab)
        for doc in corpus:
            for word in doc.split():
                if word not in self.vocab:
                    self.vocab[word] = idx
                    self.rev_vocab[idx] = word
                    idx += 1
        
    def encode(self, text):
        return [self.vocab.get(word, self.vocab['<unk>']) for word in text.split()]

    def decode(self, token_ids):
        return ' '.join([self.rev_vocab.get(idx, '<unk>') for idx in token_ids])

    def __len__(self):
        return len(self.vocab)

def prepare_batch(batch_texts, tokenizer, device):
    """Converts a list of texts to a padded tensor of token IDs."""
    batch_ids = [tokenizer.encode(text) for text in batch_texts]
    max_len = max(len(ids) for ids in batch_ids)
    
    padded_batch = torch.full((len(batch_ids), max_len), tokenizer.vocab['<pad>'], dtype=torch.long)
    for i, ids in enumerate(batch_ids):
        padded_batch[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        
    return padded_batch.to(device)
