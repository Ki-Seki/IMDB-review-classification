class Tokenizer:
    def __init__(self, num_words) -> None:
        self.num_words = num_words
        self._vocab = [''] * self.num_words
        self.word_index = [0] * self.num_words

    def fit_on_texts(self, texts: list):
        pass

    def texts2seqs(self, texts: list):
        pass