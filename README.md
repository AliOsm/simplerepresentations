# Simple Representations

This library is based on the [Transformers](https://github.com/huggingface/transformers) library by HuggingFace. Using this library, you can quickly extract text representations from Transformer models. Only two lines of code are needed to initialize the required model and extract the text representations from it.

# Table of contents

* [Installation](#installation)
	* [With `pip`](#with-pip)
	* [From source](#from-source)
* [Usage](#usage)
	* [Minimal Start](#minimal-start)
	* [Default Settings](#default-settings)
	* [Current Pretrained Models](#current-pretrained-models)
* [Acknowledgements](#acknowledgements)

## Installation

This repository is tested on Python 3.6.8 and PyTorch 1.2.0

### With `pip`

First you need to install PyTorch. Please refer to [PyTorch installation page](https://pytorch.org/get-started/locally/#start-locally) regarding the specific install command for your platform.

When PyTorch has been installed, Simple Representation can be installed using pip as follows:

```
pip install simplerepresentation
```

### From source

Here also, you first need to install PyTorch. Please refer to [PyTorch installation page](https://pytorch.org/get-started/locally/#start-locally) regarding the specific install command for your platform.

When PyTorch has been installed, you can install from source by cloning the repository and running:

```
pip install .
```

## Usage

### Minimal Start

The following example extracts the text representations from `BERT Base Uncased` model for the sentences `Hello Transformers!` and `It's very simple.`.

```python
from simplerepresentations import RepresentationModel


def load_data():
	return ['Hello Transformers!', 'It\'s very simple.']


if __name__ == '__main__':
	model_type = 'bert'
	model_name = 'bert-base-uncased'

	representation_model = RepresentationModel(
		model_type=model_type,
		model_name=model_name,
		batch_size=32,
		max_seq_length=10, # truncate sentences to be less than or equal to 10 tokens
		combination_method='cat', # concatenate the last `last_hidden_to_use` hidden states
		last_hidden_to_use=4 # use the last 4 hidden states to build tokens representations
	)

	text_a = load_data()

	all_sentences_representations, all_tokens_representations = representation_model(text_a=text_a)

	print(all_sentences_representations.shape) # (2, 768) => (number of sentences, hidden size)
	print(all_tokens_representations.shape) # (2, 10, 3072) => (number of sentences, number of tokens, hidden size)
```

You can change the code in `load_data` function to load your own data from any source you want (e.g. a CSV file).

### Default Settings

The default settings for `RepresentationModel` class are given below:

#### batch_size (32): integer
The batch size will be used while extracting representations.

#### max_seq_length (128): integer
Maximum sequence length the model will support.

#### last_hidden_to_use (1): integer
The number of the last hidden states that will be used to build the representations.

#### combination_method ('sum'): string ('sum', 'cat')
The method that will be used to combine the `last_hidden_to_use`.

#### use_cuda (True): boolean
Whether to use `CUDA` or not.

#### process_count (cpu_count() - 2 if cpu_count() > 2 else 1): integer
Number of CPU cores (processes) to use when converting examples to features. Default is (number of cores - 2) or 1 if (number of cores <= 2).

#### chunksize (500): integer
The number of chunks that the examples will be divided to when converting them to features.

### Current Pretrained Models

You can find the complete list of the current pretrained models from Transformers library [documentation](https://huggingface.co/transformers/pretrained_models.html).

## Acknowledgements

None of this would have been possible without the hard work by the HuggingFace team in developing the [Transformers](https://github.com/huggingface/transformers) library.

Also, a lot of ideas used in this repository inspired from the [Simple Transformers](https://github.com/ThilinaRajapakse/simpletransformers) library.
