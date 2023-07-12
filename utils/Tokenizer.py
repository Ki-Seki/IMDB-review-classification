class Tokenizer:
    """
    A simple tokenizer
    """

    def __init__(self, num_words) -> None:
        self.num_words = num_words  # Maximum words to be tokenized.
        self.word_index = {}  # {word: index}
        self.index_word = []  # index_word[i] == word

    def fit_on_texts(self, texts: list, 
                     filter_=lambda x: (x.islower() or x.isdigit()), 
                     split_=str.split):
        """
        Within texts, get the tokens of words with top num_words occurrences.
        :param texts: list of str texts
        :param filter_: a function filters out chars in a str
        :param split_: a function turns a str to a list of splited strs
        """
        word_count = {}
        for text in texts:
            words = split_(text)
            for word in words:
                word = word.lower()
                word = ''.join(list(filter(filter_, word)))
                if not word:
                    continue
                if word in word_count:
                    word_count[word] += 1
                else:
                    word_count[word] = 1

        word_count = dict(sorted(word_count.items(), key=lambda item: item[1], reverse=True)[:self.num_words])
        for index, word in enumerate(word_count):
            self.word_index[word] = index
            self.index_word.append(word)

    def texts2seqs(self, texts: list) -> list:
        """
        Convert the list of texts to their corresponding seqs.
        :param texts: list of str texts
        """
        pass