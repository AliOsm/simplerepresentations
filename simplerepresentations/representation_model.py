from multiprocessing import cpu_count

import torch
import numpy as np

from tqdm.auto import tqdm

from torch.utils.data import (
	DataLoader,
	SequentialSampler
)

from transformers import (
	BertConfig,       BertModel,       BertTokenizer,
	XLMConfig,        XLMModel,        XLMTokenizer,
	XLNetConfig,      XLNetModel,      XLNetTokenizer,
	RobertaConfig,    RobertaModel,    RobertaTokenizer,
	DistilBertConfig, DistilBertModel, DistilBertTokenizer,
	AlbertConfig,     AlbertModel,     AlbertTokenizer
)

from simplerepresentations.input_example import InputExample
from simplerepresentations.utils import examples_to_dataset


class RepresentationModel:
	COMBINATION_METHODS = ['sum', 'cat']
	MODELS_W_SENREP = ['bert', 'roberta', 'albert']
	WODELS_WO_SENREP = ['xlm', 'xlnet', 'distilbert']
	MODEL_CLASSES = {
		'bert':       (BertConfig,       BertModel,       BertTokenizer),
		'xlm':        (XLMConfig,        XLMModel,        XLMTokenizer),
		'xlnet':      (XLNetConfig,      XLNetModel,      XLNetTokenizer),
		'roberta':    (RobertaConfig,    RobertaModel,    RobertaTokenizer),
		'distilbert': (DistilBertConfig, DistilBertModel, DistilBertTokenizer),
		'albert':     (AlbertConfig,     AlbertModel,     AlbertTokenizer),
	}


	def __init__(
			self,
			model_type,
			model_name,
			batch_size=32,
			max_seq_length=128,
			last_hidden_to_use=1,
			combination_method='sum',
			use_cuda=True,
			process_count=cpu_count() - 2 if cpu_count() > 2 else 1,
			chunksize=500,
			verbose=1
		):
		model_type = model_type.lower()
		model_name = model_name.lower()
		combination_method = combination_method.lower()

		if combination_method not in self.COMBINATION_METHODS:
			raise ValueError('Combination method {} was not found in combination methods list ({})'.format(combination_method, ', '.join(self.COMBINATION_METHODS)))

		self.model_type = model_type
		self.model_name = model_name
		self.batch_size = batch_size
		self.max_seq_length = max_seq_length
		self.last_hidden_to_use = last_hidden_to_use
		self.combination_method = combination_method
		self.use_cuda = use_cuda
		self.process_count = process_count
		self.chunksize = chunksize
		self.verbose = verbose

		_, model_class, tokenizer_class = self.MODEL_CLASSES[model_type]

		self.model = model_class.from_pretrained(self.model_name, output_hidden_states=True)
		self.tokenizer = tokenizer_class.from_pretrained(self.model_name)

		if self.use_cuda:
			self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		else:
			self.device = torch.device('cpu')


	def __call__(self, text_a, text_b=None):
		self._move_model_to_device()

		if text_b == None:
			text_b = [None] * len(text_a)
		else:
			assert(len(text_a) == len(text_b))

		examples = [InputExample(i, text[0], text[1]) for i, text in enumerate(zip(text_a, text_b))]
		dataset = examples_to_dataset(
			examples=examples,
			tokenizer=self.tokenizer,
			max_seq_length=self.max_seq_length,
			process_count=self.process_count,
			chunksize=self.chunksize,
			verbose=self.verbose
		)
		sampler = SequentialSampler(dataset)
		dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.batch_size)

		all_sentences_representations = list()
		all_tokens_representations = list()

		for batch in tqdm(dataloader, disable=(self.verbose == 0)):
			self.model.eval()
			batch = tuple(b.to(self.device) for b in batch)

			with torch.no_grad():
				inputs = self._get_inputs_dict(batch)

				if self.model_type in self.MODELS_W_SENREP:
					_, sentences_representations, tokens_representations = self.model(**inputs)

					sentences_representations = np.array([sentences_representation.cpu().numpy() for sentences_representation in sentences_representations])
					all_sentences_representations.extend(sentences_representations)
				elif self.model_type in self.MODELS_WO_SENREP:
					_, tokens_representations = self.model(**inputs)

				tokens_representations = np.array([tokens_representation.cpu().numpy() for tokens_representation in tokens_representations]).transpose(1, 2, 0, 3)
				for tokens_representation in tokens_representations:
					if self.combination_method == 'sum':
						final_tokens_representation = np.array([np.sum(np.stack(layer)[-self.last_hidden_to_use:], 0) for layer in tokens_representation])
					elif self.combination_method == 'cat':
						final_tokens_representation = np.array([np.concatenate(tuple(layer[-self.last_hidden_to_use:])) for layer in tokens_representation])
					all_tokens_representations.append(final_tokens_representation)

		if self.model_type in self.MODELS_W_SENREP:
			return np.array(all_sentences_representations), np.array(all_tokens_representations)
		elif self.model_type in self.MODELS_WO_SENREP:
			return np.array(all_tokens_representations)


	def _get_inputs_dict(self, batch):
		inputs = {
			'input_ids':      batch[0],
			'attention_mask': batch[1]
		}

		# XLM, DistilBERT and RoBERTa don't use segment_ids
		if self.model_type != 'distilbert':
			inputs['token_type_ids'] = batch[2] if self.model_type in ['bert', 'xlnet'] else None

		return inputs


	def _move_model_to_device(self):
		self.model.to(self.device)
