import torch


class Tokenizer:
    def __init__(self):
        self.pad_token = '<PAD>'
        self.start_token= '<SOS>'
        self.end_token = '<EOS>'
        self.unk_token = '<UNK>'

        self.special_tokens = [self.pad_token, self.start_token, self.end_token, self.unk_token]
        self.char_to_idx = {token: idx for idx, token in enumerate(self.special_tokens)}
        self.idx_to_char = {idx: token for idx, token in enumerate(self.special_tokens)}
    def build_vocab(self, latex_strings):
        unique_chars = sorted(list(set(''.join(latex_strings))))
        for char in unique_chars:
            if char not in self.char_to_idx:
                idx = len(self.char_to_idx)
                self.char_to_idx[char] = idx
                self.idx_to_char[idx] = char
    def encode(self, latex_string, max_length = 128):
        tokens = [self.char_to_idx[self.start_token]] #!
        for char in latex_string:
            tokens.append(self.char_to_idx.get(char, self.char_to_idx[self.unk_token]))
        tokens.append(self.char_to_idx[self.end_token]) #!
        if len(tokens)< max_length:
            tokens += [self.char_to_idx[self.pad_token]] * (max_length - len(tokens))
        else:
            tokens = tokens[:max_length]
            tokens[-1] = self.char_to_idx[self.end_token] # Ensure the last token is <EOS>
        return torch.tensor(tokens, dtype=torch.long)
    def decode(self, token_ids):
        chars=[]
        for idx in token_ids:
            if isinstance(idx, torch.Tensor):
                idx = idx.item()

            tok = self.idx_to_char.get(idx, "")
            if tok == self.end_token or tok == self.pad_token:
                break
            if tok not in [self.start_token, self.unk_token]:
                chars.append(tok)
        return ''.join(chars)
    def vocab_size(self):
        return len(self.char_to_idx)