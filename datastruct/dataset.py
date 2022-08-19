
from multiprocessing.sharedctypes import Array
from os import device_encoding
import pandas as pd
import glob
import logging
from itertools import chain
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from transformers import PreTrainedTokenizerFast
from config import PAD, Config
# from datastruct import Instance, Span, ContextEmb
from tqdm import tqdm
# from datastruct import Sentence, Instance
from typing import List
import re
from enum import Enum

class ContextEmb(Enum):
    none = 0
    elmo = 1
    bert = 2 # not support yet
    flair = 3 # not support yet


logger = logging.getLogger(__name__)

TOKEN_ID = "input_ids"

LABEL = "labels"
GOLD_LABEL = "gold_labels"

LABELID = "labels_id"
GOLD_LABELID = "gold_labels_id"

ATTENTION_MASK = "attention_mask"
WORD_SEQ_LENS = "word_seq_lens" 
TOKEN_SEQ_LENS = "token_seq_lens"
TOKEN2WORD_ID = "token2word_ids" 
TOKEN2WORD_MASK = 'token2word_mask'
SCORES = "scores"
WEIGHTS = "weights"
ISCLEAN = "isclean"
DOMAIN = "domain"
INTENT = "intent"
ISOTHER = "isother"
ANNOTATION_MASK = 'annotation_mask'

class MapStyleJsonDataset(torch.utils.data.Dataset):
    """
    Dataset class to access ith instance.

    A sample of ith output:
    {
        'token_id': [0, 22936, 11301, 70, 5977, 2],
        'words': ['please','play','the','radio'],
        'label': ['O', 'O', 'O', 'O'],
        'gold_label': ['O', 'O', 'O', 'O'],
        'text': 'please play the radio',
        'scores': [0, 0.3, 1, 1.2],
        'weights': [0.2, 3, 1, 0.5],
        'word_seq_lens': 4
        'token_seq_lens': 6
    }

    """
    def __init__(self, glob_path: str, label2int: Dict[str, int]):

        files = glob.glob(glob_path, recursive=True)
        files.sort()
        logger.info(f"Found {len(files)} in {glob_path}.")

        reader_list = [pd.read_json(f, lines=True, chunksize=500_000) for f in files]
        reader = chain(*reader_list)
        self.df = pd.concat(reader, ignore_index=True)

        self.label2int = label2int
        self.ids = list(range(len(self.df)))

        self.updated = [{} for i in range(len(self.df))] 

    def __len__(self):
        return len(self.ids)  

    def update(self, i, key, value):
        ii = self.ids[i]
        self.updated[ii][key] = value

    def __getitem__(self, i):
        ii = self.ids[i]
        row = self.df.iloc[ii, ].to_dict()
        row[LABELID] = [self.label2int[l] for l in row[LABEL]]
        row[GOLD_LABELID]= [self.label2int[l] for l in row[GOLD_LABEL]]
        row[ISCLEAN] = [l==g for l,g in zip(row[LABEL], row[GOLD_LABEL])]
        row[ISOTHER] = [l=='O' for l in row[LABEL]]
        row[WORD_SEQ_LENS] = len(row[LABEL]) 
        row[TOKEN_SEQ_LENS] = len(row[TOKEN_ID])  

        if self.updated[ii]:
            for key in self.updated[ii].keys():
                row[key] = self.updated[ii][key]
        return row
    
    def setids(self, ids):    
        self.ids = ids


class Collator:
    """
    Collator forms a list of instances into one batch
    """
    def __init__(self, tokenizer: PreTrainedTokenizerFast, device: str, label_size = int):
        self.tokenizer = tokenizer
        self.device = device
        self.label_size = label_size

    def __call__(self, items: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        items: a list of dictionary, each dict contains one kind of data feature (batch_size, )
        """
        batch = dict()
        # fields
        for _field in [TOKEN_ID, LABELID, GOLD_LABELID, WORD_SEQ_LENS, TOKEN_SEQ_LENS]:
            field_seq = [e[_field] for e in items]
            if self.tokenizer is not None:
                batch[_field] = self._tensorize_batch(field_seq, padval = self.tokenizer.pad_token_id)
            else:
                batch[_field] = self._tensorize_batch(field_seq, padval = 0)
        
        for _field in [SCORES, WEIGHTS, ISCLEAN, INTENT, DOMAIN]:
            if _field in items[0]:
                field_seq = [e[_field] for e in items]
                batch[_field] = self._tensorize_batch(field_seq, padval = -1e10)    

        # attention_mask
        seq_len = torch.tensor([e[TOKEN_SEQ_LENS] for e in items], dtype=torch.long)
        maxlen = max(seq_len.tolist())
        batch[ATTENTION_MASK] = (torch.arange(maxlen)[None, :] < seq_len[:, None]).to(self.device)
        
        # token2word_mask
        if self.tokenizer is not None:
            word_seq_len = torch.tensor([e[WORD_SEQ_LENS] for e in items], dtype=torch.long)
            maxwordlen = max(word_seq_len.tolist())
            token2word_mask = torch.ones((len(items), maxwordlen), dtype = torch.long)*self.tokenizer.pad_token_id
            for i, e in enumerate(items):
                for j, id in enumerate(e['token2word_ids']):
                    token2word_mask[i, j] = id
            batch[TOKEN2WORD_MASK] = token2word_mask.to(self.device)  
        else:
            batch[TOKEN2WORD_MASK] = None

        # isother mask
        annotation_mask = torch.zeros((len(items), maxlen, self.label_size), dtype = torch.long)
        for i,e in enumerate(items):
            for pos in range(e['word_seq_lens']):
                if e['isother'][pos]:
                    annotation_mask[i, pos, :] = 1
                else:
                    annotation_mask[i, pos, e['labels_id'][pos]] = 1
            annotation_mask[i, e['word_seq_lens']:, :] = 1
        
        batch[ANNOTATION_MASK] = annotation_mask.to(self.device)
        
        return batch

    def _tensorize_batch(
            self, examples: List[Union[List[int], torch.Tensor, int]], padval = 0
    ) -> torch.Tensor:
        """
        examples: list of instance-associated variables, e.g. list of instance-length, or list of instance-words
        """
        # todo whenever, a dataset instance provides new example type, add the corresponding way to batch examples here
        if isinstance(examples[0], int):
            return torch.tensor(examples, dtype=torch.long).to(self.device)
        if isinstance(examples[0], (list, tuple, np.ndarray)):
            # convert List[int] to torch.tensor with Long type.
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]
        
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        
        if are_tensors_same_length:
            return torch.stack(examples, dim=0).to(self.device) # list of tensorized examples
        else:
            return pad_sequence(examples, batch_first=True, padding_value=padval).to(self.device)  # todo we probably need to mask up to 512 for HF evaluation


def batching_list_iterator(config: Config, insts: MapStyleJsonDataset, data_collator = None):
    insts_num = len(insts)
    batch_size = config.batch_size
    total_batch = insts_num // batch_size + 1 if insts_num % batch_size != 0 else insts_num // batch_size
    batched_data = []
    for batch_id in range(total_batch):
        insts_ids = range(batch_id * batch_size, min((batch_id + 1) * batch_size, insts_num))
        insts_batch = []
        for i in insts_ids:
            insts_batch.append(insts[i])    
        batched_data.append(data_collator(insts_batch))
    return batched_data


def batching_list_mask_iterator(config: Config, batched_data: List[dict], label_tag_mask):
    batched_mask = []
    start = 0
    for batch_id in range(len(batched_data)):
        word_seq_len = batched_data[batch_id]['word_seq_lens']
        batch_label_tag_mask, end = simple_batching_mask_iterator(config, word_seq_len, label_tag_mask, start)
        batched_mask.append(batch_label_tag_mask)
        start = end
    return batched_mask


def simple_batching_mask_iterator(config, word_seq_len, mask, start) -> Tuple:

    """
    batching these instances together and return tensors. 
    The seq_tensors for word and char contain their word id and char id.
    :return
        label_tag_mask: Shape: (batch_size, max_seq_length)
    """
    max_seq_len = word_seq_len.max()
    batch_size = len(word_seq_len)
    label_tag_mask = torch.zeros((batch_size, max_seq_len), dtype = torch.long)
    for idx in range(batch_size):
        label_tag_mask[idx, :word_seq_len[idx]] = torch.LongTensor(mask[start:(start+word_seq_len[idx])])
        start += word_seq_len[idx]
    end = start
    label_tag_mask = label_tag_mask.to(config.device) if label_tag_mask is not None else None
    return label_tag_mask, end