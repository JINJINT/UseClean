import json
from numpy import true_divide
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Union
import re
import logging
import argparse
import os
from transformers import AutoTokenizer, PreTrainedTokenizerFast, BertTokenizerFast
from config.config import *

logger = logging.getLogger(__name__)

DUMMY_LABEL = "DUMMY"
TOKEN_ID = "input_ids"
LABEL = "labels"
GOLD_LABEL = "gold_labels"
TEXT = "text"
WORDS = 'words'
TOKEN2WORD_ID = 'token2word_ids'
B = "B-"
I = "I-"
S = "S-"
E = "E-"
O = "O"
START_TAG = "<START>"
STOP_TAG = "<STOP>"
PAD = "<PAD>"
UNK = "<UNK>"
unk_id = 1

def convert_tag_to_BIO(tag, entity_list=None):  # todo entity_list, default_other
    """convert a tag list to BIO-scheme

    Example: ['Other', 'Other', 'Other', 'ProductSortType', 'ItemName', 'ItemName', 'ItemName'] ->
        ['O', 'O', 'O', 'B-ProductSortType', 'B-ItemName', 'I-ItemName', 'I-ItemName']
    If `entity_list` is provided, any entities not in the list are converted to `O`.
    Otherwise, entities with the values of `Other` are converted to `O`.
    """
    BIO_scheme = []
    def is_entity(x):
        return x in entity_list if entity_list else x not in ["O","DUMMY"]
    pre = None
    for cur in tag:
        if (pre is None or cur != pre) and is_entity(cur):
            BIO_scheme.append("B-" + cur)
        elif (cur == pre) and is_entity(cur):
            BIO_scheme.append("I-" + cur)
        else:
            BIO_scheme.append(cur)
        pre = cur
    
    return useiobes(BIO_scheme)

def useiobes(output: List[str]):
    """convert a tag list of BIO-scheme to BIOES-scheme
    
    Example: ['O', 'O', 'O', 'B-ProductSortType', 'B-ItemName', 'I-ItemName', 'I-ItemName'] ->
        ['O', 'O', 'O', 'S-ProductSortType', 'B-ItemName', 'I-ItemName', 'E-ItemName']
    """
    for pos in range(len(output)):
        curr_entity = output[pos]
        if pos == len(output) - 1:
            if curr_entity.startswith(B):
                output[pos] = curr_entity.replace(B, S)
            elif curr_entity.startswith(I):
                output[pos] = curr_entity.replace(I, E)
        else:
            next_entity = output[pos + 1]
            if curr_entity.startswith(B):
                if next_entity.startswith(O) or next_entity=='DUMMY' or next_entity.startswith(B):
                    output[pos] = curr_entity.replace(B, S)
            elif curr_entity.startswith(I):
                if next_entity.startswith(O) or next_entity=='DUMMY' or next_entity.startswith(B):
                    output[pos] = curr_entity.replace(I, E)
    return output                
   


def _read_txt(
        in_file: str,
        out_file: str,
        tokenizer: PreTrainedTokenizerFast,
        digit2zero: bool = True,
        config: Config=None
) -> None:
    """
    param
        in_file: what file to read in, should be of .txt format
        out_file: what file to write into, should be of .json format
        tokenizer: what tokenizer to use (this is only used for bert) 
    return
        directly write to out_file, no returns
    """
    logger.info(f"Reading from file: {in_file}; Writing to file: {out_file}")
    instance_count = 0
    with open(in_file, 'r', encoding='utf-8') as fin, open(out_file, "w") as fout:
        words = []
        labels = []
        gold_labels = []
        for line in tqdm(fin.readlines() + [""]):  # append "" to the end of `fin` indicating the end of file
            line = line.rstrip()

            if line == "":  # reach the instance break
                labels = useiobes(labels)
                gold_labels = useiobes(gold_labels)
                instance = _tokenize_instance(words, labels, gold_labels, tokenizer, config)
                if instance is None:
                    logger.info("skip empty line.")
                else:
                    fout.write(json.dumps(instance))
                    fout.write("\n")
                    instance_count += 1
                # reset
                words.clear()
                labels.clear()
                gold_labels.clear()
                continue

            if len(line.split()) == 1:  # one columns todo check with Jinjin what a single line represents for
                word = ','
                label = line.split()[0]
                gold_label = label
            else:
                if len(line.split()) == 2:  # two columns
                    # !!! we assume that test and dev has only two columns and the label is gloden label
                    word, label = line.split()
                    gold_label = label
                elif len(line.split()) == 3:  # three columns
                    word, label, gold_label = line.split()

            if digit2zero:
                word = re.sub(r'\d', '0', word)  # replace digit with 0.

            words.append(word)
            labels.append(label)
            gold_labels.append(gold_label)

    logger.info(f"number of sentences: {instance_count}")


def _tokenize_instance(
        words: List[str],
        labels: List[str],
        gold_labels: List[str],
        tokenizer: PreTrainedTokenizerFast,
        config: Config
) -> Union[Dict, None]:
    """
    this function is to tokenize and pack the needed information for this utterance
    params:
        words: list of words in this utterance
        labels: list of observed labels for this utterance
        gold_labels: list of gold labels for this utterance
        tokenizer: a tokenizer for tokenization, will be set as None for bilstm encoder
        config: the object in class config 
    return:
        a dictionary contains all the info for this utterance we needed    
    """
    assert (len(words) == len(labels)) and (len(words) == len(gold_labels)), \
        "words, labels and gold labels should have the same lengths."
    if len(words) == 0:  # skip empty instance
        return None

    # tokenization
    if tokenizer is not None:
        token_encoding = tokenizer(words, is_split_into_words=True)
        token_to_word = []
        for i, (l, g_l) in enumerate(zip(labels, gold_labels)):
            token_span = token_encoding.word_to_tokens(i)
            token_to_word.append(token_span.start)
        return {
            TOKEN_ID: token_encoding["input_ids"],
            TOKEN2WORD_ID: token_to_word,
            WORDS: words,
            LABEL: labels,
            GOLD_LABEL: gold_labels,
            TEXT: " ".join(words),
        }
    else:
        tokenid = [config.word2idx[word] for word in words]
        return {
            TOKEN_ID: tokenid,
            TOKEN2WORD_ID: None,
            WORDS: words,
            LABEL: labels,
            GOLD_LABEL: gold_labels,
            TEXT: " ".join(words),
        }



def build_word_idx(config) -> None:
    """
    Build the vocab 2 idx for all utterances
    """
    words = []
    with open('../data/'+config.dataset+'/train.txt', 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            line = line.rstrip()
            if len(line.split())>0:
                word = line.split()[0]
                if config.digit2zero:
                    word = re.sub(r'\d', '0', word)  # replace digit with 0.
                words.append(word) 
    f.close()

    with open('../data/'+config.dataset+'/test.txt', 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            line = line.rstrip()
            if len(line.split())>0:
                word = line.split()[0]
                if config.digit2zero:
                    word = re.sub(r'\d', '0', word)  # replace digit with 0.
                words.append(word) 
    f.close()

    with open('../data/'+config.dataset+'/dev.txt', 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            line = line.rstrip()
            if len(line.split())>0:
                word = line.split()[0]
                if config.digit2zero:
                    word = re.sub(r'\d', '0', word)  # replace digit with 0.
                words.append(word) 
    f.close()
    
    config.word2idx = dict()
    config.idx2word = []
    config.word2idx[PAD] = 0
    config.idx2word.append(PAD)
    config.word2idx[UNK] = 1
    config.unk_id = 1
    config.idx2word.append(UNK)

    config.char2idx[PAD] = 0
    config.idx2char.append(PAD)
    config.char2idx[UNK] = 1
    config.idx2char.append(UNK)

    # extract char on train, dev, test
    for word in words:
        if word not in config.word2idx:
            config.word2idx[word] = len(config.word2idx)
            config.idx2word.append(word)
    
    # extract char only on train (doesn't matter for dev and test)
    for word in words:
        for c in word:
            if c not in config.char2idx:
                config.char2idx[c] = len(config.idx2char)
                config.idx2char.append(c)


def build_emb_table(config) -> None:
    """
    for bilstm encoder:
    build the embedding table with pretrained word embeddings if given otherwise, use random embeddings
    """
    count=0
    print("Building the embedding table for vocabulary...")
    scale = np.sqrt(3.0 / config.embedding_dim)
    if config.embedding is not None:
        print("[Info] Use the pretrained word embedding to initialize: %d x %d" % (len(config.word2idx), config.embedding_dim))
        config.word_embedding = np.empty([len(config.word2idx), config.embedding_dim])
        for word in config.word2idx:
            if word in config.embedding:
                config.word_embedding[config.word2idx[word], :] = config.embedding[word]
            elif word.lower() in config.embedding:
                config.word_embedding[config.word2idx[word], :] = config.embedding[word.lower()]
            else:
                count+=1
                config.word_embedding[config.word2idx[word], :] = np.random.uniform(-scale, scale, [1, config.embedding_dim])
        config.embedding = None
    else:
        config.word_embedding = np.empty([len(config.word2idx), config.embedding_dim])
        for word in config.word2idx:
            config.word_embedding[config.word2idx[word], :] = np.random.uniform(-scale, scale, [1, config.embedding_dim])
    print(count)


def read_pretrain_embedding(config) -> Tuple[Union[Dict[str, np.array], None], int]:
    """
    Read the pretrained word embeddings, return the complete embeddings and the embedding dimension
    """
    print("reading the pretraing embedding: %s" % (config.embedding_file))
    if config.embedding_file is None:
        print("pretrain embedding in None, using random embedding")
        config.embedding = None
        return 
    else:
        exists = os.path.isfile(config.embedding_file)
        if not exists:
            print(colored("[Warning] pretrain embedding file not exists, using random embedding",  'red'))
            config.embedding = None
            return
            
    embedding_dim = -1
    embedding = dict()
    remove=[]
    with open(config.embedding_file, 'r', encoding='utf-8') as file:
        file.readline()#for spanish embedding
        for line in tqdm(file.readlines()):
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if embedding_dim < 0:
                embedding_dim = len(tokens) - 1
            else:
                
                if(embedding_dim + 1 != len(tokens)):       
                    remove.append(tokens[0])
                    continue
            embedd = np.empty([1, embedding_dim])
            embedd[:] = tokens[1:]
            first_col = tokens[0]
            embedding[first_col] = embedd
    config.embedding = embedding
    config.embedding_dim = embedding_dim
    
