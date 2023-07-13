import unittest


from utils.Tokenizer import Tokenizer

class InitTest(unittest.TestCase):
    def testNumIs100(self):
        tk = Tokenizer(100)
        self.assertEqual(tk.num_words, 100)
        self.assertEqual(tk.word_index, {})
        self.assertEqual(tk.index_word, [])


class FitTest(unittest.TestCase):
    def __testWord_indexAndIndex_word(self, tk: Tokenizer, length: int):
        """
        An assistant for rest test functions. word_index and index_word 
        should match each other.
        """
        self.assertEqual(len(tk.word_index), length)
        self.assertEqual(len(tk.index_word), length)
        for i in range(length):
            self.assertEqual(tk.word_index.get(tk.index_word[i], -1), i+1)

    def testDupInASentence(self):
        texts = ['I love you love me！']
        tk = Tokenizer(10)
        tk.fit_on_texts(texts)
        self.__testWord_indexAndIndex_word(tk, 4)  # tokens: I, love, you, me

    def testDupInASentenceButSmallNum_words(self):
        texts = ['I love you love me！']  # after split: I, love, you, me, !
        tk = Tokenizer(3)
        tk.fit_on_texts(texts)
        self.assertIn('love', tk.word_index)  # love must be in the word_index
        self.__testWord_indexAndIndex_word(tk, 3)

    def testDupAmongSentences(self):
        texts = ['I love you love me！', 
                 'HOW to find love is a question.',
                 'this Is only 1 test...',
                 '$%^&*( 09*()) (* +-/+_!~!@~`1'')']
        tk = Tokenizer(10)
        tk.fit_on_texts(texts)
        self.assertIn('love', tk.word_index)  # love must be in the word_index
        self.assertIn('i', tk.word_index)
        self.__testWord_indexAndIndex_word(tk, 10)
    
    def testAllLowerCaseAndNoMarks(self):
        texts = ['I love you love me！', 
                 'HOW to find love is a question.',
                 'this is only 1 test...',
                 '$%^&*( 09*()) cool (* +-/+_!~!@~`1'')']
        tk = Tokenizer(1000)
        tk.fit_on_texts(texts)
        for word in tk.index_word:
            for ch in word:
                self.assertTrue(ch.islower() or ch.isdigit())


class Texts2SeqsTest(unittest.TestCase):
    def testNormalCase(self):
        texts = ['I love you love me！', 
                 'HOW to find love is a question.',
                 'this is only 1 test...',
                 '$%^&*( 09*()) cool (* +-/+_!~!@~`1'')']
        tk = Tokenizer(1000)
        tk.fit_on_texts(texts)
        seqs = tk.texts2seqs(['Do you love me?', 'Yes!', '1 is one', '2 is two'])

        self.assertEqual(len(seqs), 4)

        self.assertEqual(len(seqs[0]), 3)
        self.assertEqual(len(seqs[1]), 0)
        self.assertEqual(len(seqs[2]), 2)
        self.assertEqual(len(seqs[3]), 1)

        # love is the most frequent , so its index must be 0
        self.assertEqual(seqs[0][1], 1)

    def testPadding(self):
        texts = ['I love you love me！', 
                 'HOW to find love is a question.',
                 'this is only 1 test...',
                 '$%^&*( 09*()) cool (* +-/+_!~!@~`1'')']
        tk = Tokenizer(1000)
        tk.fit_on_texts(texts)
        seqs = tk.texts2seqs(['Do you love me?', 'Yes!', 
                              '1 is one', '2 is two'], 
                              seq_len=5, padding_val=-2)
        self.assertEqual(len(seqs), 4)
        for i in range(4):
            self.assertEqual(len(seqs[i]), 5)
            self.assertEqual(seqs[i][0], -2)
            self.assertEqual(seqs[i][1], -2)

if __name__ == '__main__':
    unittest.main()