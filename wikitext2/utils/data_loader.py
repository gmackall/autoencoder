import torch
from torchtext import data
from torchtext import datasets
import os.path
from torchtext.datasets import WikiText2
import spacy

from spacy.symbols import ORTH
my_tok = spacy.load('en')

def spacy_tok(x):
    return [tok.text for tok in my_tok.tokenizer(x)]

def download_data(batch_size, data_root):
    return datasets.WikiText2.iters(batch_size=batch_size, root=data_root, device=torch.device)

def load_data(batch_size, data_root="./datasets/data"):
    assert os.path.exists(data_root), "Bad data_root supplied"
    return datasets.WikiText2.iters(batch_size=batch_size, root=data_root, device=torch.device)

def alt_load_data(batch_size, seq_length=50):
    TEXT = data.Field(lower=True, tokenize=spacy_tok)
    train, valid, test = WikiText2.splits(TEXT)
    TEXT.build_vocab(train, vectors="glove.6B.200d")
    train_iter, valid_iter, test_iter = data.BPTTIterator.splits(
        (train, valid, test),
        batch_size=batch_size,
        bptt_len=seq_length,
        device=torch.device)
    return (train_iter, valid_iter, test_iter), TEXT
        
#maybe methods for getting text
