# PyRusToken


***This repo is still under development***

PyRusToken is BPE tokenizer library that accelerates Python with Rust, providing efficient and high-performance tokenization capabilities.

# Project Scope

This project serves as a helper for tokenizing text efficiently for another project of mine, which consists in pretraining a GPT implementation from scratch and finally Mistral.

The goal is to maximize the resources available, as pretraning these exact models are virtually impossible on consumer grade hardware.



## Features

- Python & Rust Integration: Combines the simplicity of Python with the speed and safety of Rust.

- Parallelism with [Rayon](https://github.com/rayon-rs/rayon).

- Byte Pair Encoding (BPE): Implements Byte Pair Encoding tokenization for effective subword tokenization.


- Easy-to-Use Python Interface: Offers a simple Python API for seamless integration into existing projects.


## Pending Work

- Profile and find where the code takes the longest time and resources.
- Find optimizations with datatypes, e.g replacing HashMap with FxHashMap.
- Evaluate and compare the speedup with the following:
    - Python vs Rust Implementation
    - Sequential vs Parallel
    - Rayon vs std::thread

## Usage
You can test the following code in a Jupyter Notebook, or use the one in the examples folder
```python

from tokenizer import tokenizer

tok = tokenizer.BPE()


encoded = tok.encode("hello")

print(encoded)

# Output: [104, 101, 108, 108, 111]

lipsum = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed quis metus a nisl vestibulum tristique. Phasellus vel nulla non quam interdum convallis."
tok.parallel_word_tokenizer(lipsum)

# Output: ['Lorem', 'ipsum', 'dolor', 'sit', 'amet', 'consectetur', 'adipiscing', 'elit', 'Sed', 'quis', 'metus', 'a', 'nisl', 'vestibulum', 'tristique', 'Phasellus', 'vel', 'nulla', 'non', 'quam', 'interdum', 'convallis']
 

vocab_size = 276 # the desired final vocabulary size
num_merges = vocab_size - 256

tok.learn_ids(lipsum,num_merges)

print(tok.decode(tok.encode("hello world")))

# Output: 'hello world'
```

