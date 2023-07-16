import unittest

import torch

from utils.Embedding import GloVeEmbedding

class BasicTest(unittest.TestCase):

    def testOnANormalExample(self):
        path = './resources/glove.6B'
        vocab_size = 4
        embedding_dim = 100
        padding_idx = 0
        word_index = {
            '': 0,  # placeholder
            'i': 1,
            'love': 2,
            'you': 3,
            '123': 4,  # digit is allowed
            }
        input_ = torch.tensor([0, 1, 2, 3, 4])
        ebd = GloVeEmbedding(
            path, vocab_size, embedding_dim, padding_idx, word_index)
        output = ebd(input_)

        self.assertEqual(output.shape, torch.Size([vocab_size+1, embedding_dim]))

        self.assertTrue(torch.equal(output[0], torch.zeros(embedding_dim)))
        
        self.assertFalse(torch.equal(output[1], torch.zeros(embedding_dim)))
        self.assertFalse(torch.equal(output[2], torch.zeros(embedding_dim)))
        self.assertFalse(torch.equal(output[3], torch.zeros(embedding_dim)))
        self.assertFalse(torch.equal(output[4], torch.zeros(embedding_dim)))

    def testWithBadWord_index(self):
        path = './resources/glove.6B'
        vocab_size = 5
        embedding_dim = 100
        padding_idx = 0
        word_index = {
            'the': 2,  # normal
            '31': 1,  # a digit
            '#': 3,  # a mark
            '-': 0,  # index is zero
            'test': 5,  # normal
            'LoVe': 4,  # uppercase letters
            'over': 6,  # 6 > vocab_size
            }
        input_ = torch.tensor([0, 1, 2, 3, 4, 5])
        ebd = GloVeEmbedding(
            path, vocab_size, embedding_dim, padding_idx, word_index)
        output = ebd(input_)

        self.assertEqual(output.shape, torch.Size([vocab_size+1, embedding_dim]))

        self.assertTrue(torch.equal(output[0], torch.zeros(embedding_dim)))
        self.assertTrue(torch.equal(output[4], torch.zeros(embedding_dim)))
        
        self.assertFalse(torch.equal(output[1], torch.zeros(embedding_dim)))
        self.assertFalse(torch.equal(output[2], torch.zeros(embedding_dim)))
        self.assertFalse(torch.equal(output[3], torch.zeros(embedding_dim)))
        self.assertFalse(torch.equal(output[5], torch.zeros(embedding_dim)))


if __name__ == '__main__':
    unittest.main()
