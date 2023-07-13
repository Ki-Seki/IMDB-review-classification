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
            self.word_index[word] = index + 1
            self.index_word.append(word)

    def texts2seqs(self, texts: list, 
                   filter_=lambda x: (x.islower() or x.isdigit()), 
                   split_=str.split, 
                   seq_len: int=None, padding_val: int=0) -> list:
        """
        Convert the list of texts to their corresponding seqs.
        :param texts: list of str texts
        :param seq_len: maximum length of each seq; None if padding of no need
        :param padding_val: insert padding_val in front of seq if its len < seq_len

        Padding examples:

        [[1], [2, 3], [4, 5, 6]]
        -> seq_len==3, padding_val==0 ->
        [[0, 0, 1], [0, 2, 3], [4, 5, 6]]

        [[1], [2, 3], [4, 5, 6]]
        -> seq_len==2, padding_val==-2 ->
        [[-2, 1], [2, 3], [5, 6]]
        """
        # Convert to sequences
        seqs = []
        for text in texts:
            seq = []
            words = split_(text)
            for word in words:
                word = word.lower()
                word = ''.join(list(filter(filter_, word)))
                index = self.word_index.get(word, None)
                if index is not None:
                    seq.append(index)
            seqs.append(seq)

        # Padding and truncation
        for i in range (len(seqs)):
            l = len(seqs[i])
            if seq_len != None and l < seq_len:
                seqs[i] = [padding_val] * (seq_len-l) + seqs[i]
            elif seq_len != None and l > seq_len:
                seqs[i] = seqs[i][l-seq_len:]
        return seqs