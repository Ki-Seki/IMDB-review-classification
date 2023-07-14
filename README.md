# IMDB-review-classification

This project solves the IMDB review classification problem, which is a case study of [*Deep Learning with Python*](https://www.manning.com/books/deep-learning-with-python) (See section 6.1.3).

The book has an implementaion in Keras. I re-implement it using PyTorch. 

```text
.
├── resources                    # Downloaded resources
│   ├── aclImdb
│   └── glove.6B
├── tests                        # Unit tests
│   ├── test_utils_dataset.py
│   └── test_utils_tokenizer.py
└── utils
│   ├── Dataset.py               # IMDBDataset class
│   └── Tokenizer.py             # A simple tokenizer
├── .gitignore
├── LICENSE
├── main.ipynb                   # Model
├── README.md
├── requirements.txt
└── setup.sh
```

## Usage

1. Prepare the environment specified in [requirements.txt](./requirements.txt).
2. Run [setup.sh](./setup.sh) to prepare the requested resources (IMDB and GloVe).
3. Run [main.py](main.py).

## What I've learned

* PyTorch development life-cycle
* TDD (Test Driven Development) practice
* Tokenizer implementation (because there is no tokenizer in PyTorch as easy as Keras' tokenizer)
* IMDB dataset preprocessing
* GloVe embedding usage

## TODO

- [ ] The tokenizer and sequence padding should be seperated.
- [ ] Tokenizer should support using library.