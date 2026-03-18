import pandas as pd
import torch
from torch.utils.data import Dataset
from .vocab import Vocabulary, encode_sentence

class TranslationDataset(Dataset):
    def __init__(self, df, en_vocab, hi_vocab, max_len=50):
        self.en_sentences = df["en"].tolist()
        self.hi_sentences = df["hi"].tolist()
        self.en_vocab = en_vocab
        self.hi_vocab = hi_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.en_sentences)

    def __getitem__(self, idx):
        src = encode_sentence(self.en_sentences[idx], self.en_vocab, self.max_len)
        tgt = encode_sentence(self.hi_sentences[idx], self.hi_vocab, self.max_len)
        return torch.tensor(src), torch.tensor(tgt)

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = torch.stack(src_batch)
    tgt_batch = torch.stack(tgt_batch)
    tgt_input = tgt_batch[:, :-1]
    tgt_output = tgt_batch[:, 1:]
    return src_batch, tgt_input, tgt_output

def load_and_preprocess_data(data_path='data/English-Hindi.tsv'):
    df = pd.read_csv(data_path, sep='\t', header=None, names=["id1", "en", "id2", "hi"])
    df = df[["en", "hi"]]
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    en_vocab = Vocabulary(freq_threshold=2)
    hi_vocab = Vocabulary(freq_threshold=2)
    en_vocab.build_vocab(df["en"].tolist())
    hi_vocab.build_vocab(df["hi"].tolist())
    return df, en_vocab, hi_vocab
