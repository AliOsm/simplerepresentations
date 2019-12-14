from multiprocessing import Pool

import torch

from tqdm.auto import tqdm
from torch.utils.data import TensorDataset

from simplerepresentations.input_features import InputFeatures


def examples_to_dataset(examples, tokenizer, max_seq_length, process_count, chunksize, verbose=1):
    """
    Converts a list of InputExample objects to a TensorDataset containing InputFeatures.
    """

    tokenizer = tokenizer

    if verbose == 1:
        print('Converting to features started.')

    features = convert_examples_to_features(
        examples=examples,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        process_count=process_count,
        chunksize=chunksize,
        # XLNet has a CLS token at the end
        cls_token_at_end=bool('bert' in ['xlnet']),
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=2 if 'bert' in ['xlnet'] else 0,
        sep_token=tokenizer.sep_token,
        # RoBERTa uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
        sep_token_extra=bool('bert' in ['roberta']),
        # PAD on the left for XLNet
        pad_on_left=bool('bert' in ['xlnet']),
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=4 if 'bert' in ['xlnet'] else 0,
        verbose=verbose
    )

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

    return dataset


def convert_examples_to_features(
        examples,
        max_seq_length,
        tokenizer,
        process_count,
        chunksize,
        cls_token_at_end=False,
        sep_token_extra=False,
        pad_on_left=False,
        cls_token='[CLS]',
        sep_token='[SEP]',
        pad_token=0,
        sequence_a_segment_id=0,
        sequence_b_segment_id=1,
        cls_token_segment_id=1,
        pad_token_segment_id=0,
        mask_padding_with_zero=True,
        verbose=1
    ):
    """
    Loads a data file into a list of `InputBatch`s

    `cls_token_at_end` define the location of the CLS token:
        - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
        - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]

    `cls_token_segment_id` define the segment id associated to the CLS token:
        - 0 for BERT
        - 2 for XLNet
    """

    examples = [
        (
            example,
            max_seq_length,
            tokenizer,
            cls_token_at_end,
            cls_token,
            sep_token,
            cls_token_segment_id,
            pad_on_left,
            pad_token_segment_id,
            sep_token_extra
        ) for example in examples
    ]

    with Pool(process_count) as p:
        features = list(
            tqdm(
                p.imap(
                    convert_example_to_feature,
                    examples,
                    chunksize=chunksize
                ),
                total=len(examples),
                disable=(verbose == 0)
            )
        )

    return features


def convert_example_to_feature(
        example_row,
        pad_token=0,
        sequence_a_segment_id=0,
        sequence_b_segment_id=1,
        cls_token_segment_id=1,
        pad_token_segment_id=0,
        mask_padding_with_zero=True,
        sep_token_extra=False
    ):
    example, \
    max_seq_length, \
    tokenizer, \
    cls_token_at_end, \
    cls_token, \
    sep_token, \
    cls_token_segment_id, \
    pad_on_left, \
    pad_token_segment_id, \
    sep_token_extra = example_row

    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with '- 3'. '- 4' for RoBERTa.
        special_tokens_count = 4 if sep_token_extra else 3
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
    else:
        # Account for [CLS] and [SEP] with '- 2' and with '- 3' for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens_a) > max_seq_length - special_tokens_count:
            tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids:   0   0   0   0  0     0   0
    #
    # Where 'type_ids' are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the 'sentence vector'. Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = tokens_a + [sep_token]
    segment_ids = [sequence_a_segment_id] * len(tokens)

    if tokens_b:
        tokens += tokens_b + [sep_token]
        segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

    if cls_token_at_end:
        tokens = tokens + [cls_token]
        segment_ids = segment_ids + [cls_token_segment_id]
    else:
        tokens = [cls_token] + tokens
        segment_ids = [cls_token_segment_id] + segment_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
        segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
    else:
        input_ids = input_ids + ([pad_token] * padding_length)
        input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids
    )


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.

    while True:
        total_length = len(tokens_a) + len(tokens_b)

        if total_length <= max_length: break

        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
