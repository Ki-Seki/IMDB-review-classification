import os

import torch
from torch.nn import Module, Embedding


class GloVeEmbedding(Module):
    """
    A custom embedding layer. It's weights use the pretrained word embeddings, 
    GloVe. The size of the embedding layer is vocab_size+1 due to the 
    placeholder embedding[0] = [0 ... 0]

    Example:

    >>> word_index = {
                '': 0,  # placeholder
                'i': 1,
                'love': 2,
                'you': 3,
                '123': 4,  # digit is allowed
                }  # This is usually generated by utils.Tokenizer.word_index
    >>> ebd_layer = GloVeEmbedding(path_to_glove, 3, 50, 0, word_index)
    >>> input_ = torch.tensor([0, 1, 2])
    >>> output = ebd_layer(input_)
    >>> output.shape
    torch.Size([6, 50])
    """
    
    def __init__(self, path: str, vocab_size: int, 
                 embedding_dim: int, padding_idx: int,
                 word_index: dict):
        """
        :param path: The dir stores the GloVe files
        :param vocab_size: Size of the dictionary of embeddings
        :param embedding_dim: Size of embedding, one of {50, 100, 200, 300}
        :param padding_idx:  The embedding vector at `padding_idx` is 0
        :param word_index: {word: index} generated by utils.Tokenizer.word_index
        """
        super().__init__()
        self.path = path
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.word_index = word_index

        embedding_index = self._parse_glove_file()
        embedding_matrix = self._prepare_embedding_matrix(embedding_index)
        self.embedding_layer = Embedding.from_pretrained(
            embedding_matrix, freeze=True, padding_idx=self.padding_idx)

    def forward(self, x):
        return self.embedding_layer(x)

    def _parse_glove_file(self) -> dict:
        """Extract embeddings from a GloVe file"""
        embedding_index = {}
        ebd_path = os.path.join(self.path, f'glove.6B.{self.embedding_dim}d.txt')
        with open(ebd_path) as f:
            for line in f:
                values = line.split()
                word = values[0]
                embedding_vector = [float(ebd) for ebd in values[1:]]
                embedding_vector = torch.tensor(embedding_vector, dtype=torch.float)
                embedding_index[word] = embedding_vector
        return embedding_index
    
    def _prepare_embedding_matrix(self, embedding_index) -> torch.Tensor:
        """Convert embedding_index to embedding_matrix using word_index"""
        embedding_matrix = torch.zeros(self.vocab_size + 1, self.embedding_dim)
        # `self.vocab_size + 1` because we have embedding_matrix[0] as a placeholder

        for word, i in self.word_index.items():
            if i == 0 or i > self.vocab_size:
                # We only care about indices in [1, vocab_size]
                # i == 0 is a placeholder, word_index[''] == 0
                continue
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix
