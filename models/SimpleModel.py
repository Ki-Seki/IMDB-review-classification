from torch import nn

class SimpleModel(nn.Module):

    def __init__(self, seq_len, vocab_size, embedding_dim, embedding=None):
        super().__init__()
        self.embedding = embedding or nn.Embedding(vocab_size+1, embedding_dim, 0)
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(seq_len*embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, X):
        X = self.embedding(X)
        X = self.flatten(X)
        X = self.linear_stack(X)
        return X
