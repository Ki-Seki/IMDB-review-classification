import unittest
import os

import torch
from torch.utils.data import DataLoader, random_split

from utils.Dataset import IMDBDataset


class BasicTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(BasicTest, self).__init__(*args, **kwargs)

        # ! REFACTOR the imdb_dir if you need to
        imdb_dir = './resources/aclImdb'
        self.train_dir = os.path.join(imdb_dir, 'train')
        self.test_dir = os.path.join(imdb_dir, 'test')

    def testTrainValSplit(self):
        dataset = IMDBDataset(self.train_dir, 100, 10000)

        torch.manual_seed(123)
        train, valid, _ = random_split(dataset, [200, 10000, len(dataset)-10200])  # split problem
        self.assertEqual(len(train), 200)
        self.assertEqual(len(valid), 10000)
        data1_X, data1_y = train[5]

        torch.manual_seed(123)
        train, valid, _ = random_split(dataset, [200, 10000, len(dataset)-10200])
        self.assertEqual(len(train), 200)
        self.assertEqual(len(valid), 10000)
        data2_X, data2_y = train[5]

        self.assertTrue(torch.equal(data1_X, data2_X))
        self.assertEqual(data1_y, data2_y)

    def testUseInDataLoader(self):
        seq_len = 100
        vocab_size = 10000
        batch_size = 32
        dataset = IMDBDataset(self.train_dir, seq_len, vocab_size)
        torch.manual_seed(123)
        train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        seqs, labels = next(iter(train_dataloader))
        self.assertEqual(seqs.size(), torch.Size([batch_size, seq_len]))
        self.assertEqual(labels.size(), torch.Size([batch_size, 1]))


    def testTokenizerReuse(self):
        train_dataset = IMDBDataset(self.train_dir, 100, 10000)

        # Reuse the tokenizer in train_dataset
        test_dataset = IMDBDataset(self.test_dir, 100, 10000, 
                                   train_dataset.tokenizer)

        self.assertEqual(train_dataset.tokenizer.index_word, test_dataset.tokenizer.index_word)
        self.assertDictEqual(train_dataset.tokenizer.word_index, test_dataset.tokenizer.word_index)


if __name__ == '__main__':
    unittest.main()