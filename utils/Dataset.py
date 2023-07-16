import os

import torch
from torch.utils.data import Dataset

from utils.Tokenizer import Tokenizer


class IMDBDataset(Dataset):

    def __init__(self, path: str, seq_len: int, vocab_size: int,
                 tokenizer: Tokenizer=None):
        """
        :param path: Dataset dir
        :param seq_len: Max len of a seq
        :param vocab_size: Size of the tokenizer vocabulary
        :param tokenizer: For tokenizer reuse
        """
        self.path = path
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer

        # Preprocessing
        texts, labels = self._get_raw_data()
        seqs = self._tokenization(texts)

        self.seqs = torch.tensor(seqs)
        self.labels = torch.tensor(labels, dtype=torch.float).unsqueeze(1)
        # PS. Though labels are either 1 or 0, it is recommended to use 
        # `torch.float`, because, for example, nn.BCELoss() only supports 
        # calculations between two `torch.float`.
    
    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, idx):
        return [self.seqs[idx], self.labels[idx]]

    def _get_raw_data(self):
        """Read raw data from IMDB files"""
        texts = []
        labels = []
        for label_type in ['neg', 'pos']:
            dir_name = os.path.join(self.path, label_type)
            for fname in os.listdir(dir_name):
                if not fname.endswith('.txt'):
                    continue
                with open(os.path.join(dir_name, fname)) as f:
                    texts.append(f.read())
                labels.append(0 if label_type == 'neg' else 1)
        return texts, labels
    
    def _tokenization(self, texts):
        """
        Conduct tokenization using self.tokenizer if it's None,
        or using a new tokenizer fitted on the `texts`.
        :param texts: They will be tokenized
        :return: Sequences after tokenization
        """
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(self.vocab_size)
            self.tokenizer.fit_on_texts(texts)
        return self.tokenizer.texts2seqs(texts, seq_len=self.seq_len)
