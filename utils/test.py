import unittest
from Tokenizer import Tokenizer

class InitTest(unittest.TestCase):
    def numis100(self):
        tk = Tokenizer(100)
        self.assertEqual(tk.num_words, 100)
        self.assertEqual(len(tk._vocab), 100)

class FitTest(unittest.TestCase):
    def dupInASentence(self):
        tk = Tokenizer(['I love you love me！'])
        self.assertEqual(tk._vocab, ['i', 'love', 'you', 'me', '!'])

    def dupAmongSentences(self):
        pass

class Texts2SeqsTest(unittest.TestCase):
    def normalCase(self):
        ['I love you love me！']
        pass

if __name__ == '__main__':
    unittest.main()