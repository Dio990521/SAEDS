from torch.utils.data import Dataset
import torch

class Seq2SeqDataLoader(Dataset):
    def __init__(self, X, y, pad_len, word2id, max_size=None, glove_tokenizer=None):
        self.source = X
        self.target = y
        self.pad_len = pad_len
        self.start_int = word2id['<s>']
        self.eos_int = word2id['</s>']
        self.pad_int = word2id['<pad>']
        if max_size is not None:
            self.source = self.source[:max_size]
            self.target = self.target[:max_size]
            self.tag = self.tag[:max_size]

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        # for src add <s> ahead
        src = self.source[idx]
        src_len = len(src)
        if len(src) > self.pad_len:
            src = src[:self.pad_len]
            src_len = self.pad_len
        else:
            src = src + [self.pad_int] * (self.pad_len - len(src))

        # for trg add <s> ahead and </s> end
        trg = self.target[idx]
        if len(trg) > self.pad_len - 2:
            trg = trg[:self.pad_len-2]
        trg = [self.start_int] + trg + [self.eos_int] + [self.pad_int] * (self.pad_len - len(trg) - 2)
        if not len(src) == len(trg) == self.pad_len:
            print(src, trg)
        assert len(src) == len(trg) == self.pad_len

        return torch.tensor(src), torch.tensor([src_len]), torch.tensor(trg)
    
    def build_glove_ids(self, X):
        if glove_tokenizer is not None:
            for src in X:
                glove_id, glove_id_len = glove_tokenizer.encode_ids_pad(src)
                self.glove_ids.append(glove_id)
                self.glove_ids_len.append(glove_id_len)