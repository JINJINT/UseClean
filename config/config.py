from distutils.command.config import config
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict, Union
import torch
from enum import Enum
import os

from termcolor import colored


START = "<START>"
STOP = "<STOP>"
PAD = "<PAD>"


class Config:
    def __init__(self, args) -> None:
        """
        Construct the arguments and some hyperparameters
        :param args:
        """

        # Predefined label string.
        self.PAD = PAD
        self.B = "B-"
        self.I = "I-"
        self.S = "S-"
        self.E = "E-"
        self.O = "O"
        self.START_TAG = START
        self.STOP_TAG = STOP
        self.UNK = "<UNK>"
        self.unk_id = -1

        # Model hyper parameters
        self.embedding_file = args.embedding_file
        self.embedding_dim = args.embedding_dim
        # self.context_emb = ContextEmb[args.context_emb]
        self.context_emb_size = 0
        self.word_embedding = None
        self.seed = args.seed
        self.digit2zero = args.digit2zero
        self.hidden_dim = args.hidden_dim
        self.use_brnn = True
        self.num_layers = 1
        self.dropout = args.dropout
        self.char_emb_size = 25
        self.charlstm_hidden_dim = 50
        self.use_char_rnn = args.use_char_rnn
        self.info = args.info
        self.clmethod = args.clmethod
        self.score = args.score
        self.usescore = args.usescore
        self.modify = args.modify
        self.cutoff = args.cutoff
        self.crossfolds = args.crossfolds
        self.encoder = args.encoder
        self.cleanprop = args.cleanprop
        self.contrastive = args.contrastive
        self.lamb = args.lamb
        self.model_nums = args.model_nums
        self.alpha = args.alpha
        self.alpha_warmup_ratio = args.alpha_warmup_ratio
        self.classifier = args.classifier
        self.cleanepochs = args.cleanepochs
        self.warm = args.warm
        self.tau = args.tau
        self.usecleanscore = args.usecleanscore
        self.injectclean = args.injectclean
        self.weight = args.weight
        self.random = args.random
        self.usef1 = args.usef1
        self.recall = args.recall
        self.fakeq = args.fakeq
        self.numfake = args.numfake

        # Data specification
        self.dataset = args.dataset
        self.train_file = "./data/" + self.dataset + "/train.txt"
        self.dev_file = "./data/" + self.dataset + "/dev.txt"
        self.test_file = "./data/" + self.dataset + "/test.txt"
        self.label2idx = {}
        self.idx2label = {}
        self.char2idx = {}
        self.idx2char = []
        self.num_char = 0
        self.train_num = args.train_num
        self.dev_num = args.dev_num
        self.test_num = args.test_num
        self.num_folds = 2

        # Training hyperparameter
        self.model_folder = args.model_folder
        self.res_folder=args.res_folder
        self.optimizer = args.optimizer.lower()
        self.learning_rate = args.learning_rate
        self.momentum = args.momentum
        self.l2 = args.l2
        self.num_epochs = args.num_epochs
        self.use_dev = True
        self.batch_size = args.batch_size
        self.clip = 5
        self.lr_decay = args.lr_decay
        self.device = torch.device(args.device)
        self.num_outer_iterations = args.num_outer_iterations
        self.neg_noise_rate=args.neg_noise_rate
        self.pos_noise_rate=args.pos_noise_rate
        self.warm_up_num=args.warm_up_num
        self.num_gradual_neg=args.num_gradual_neg
        self.num_gradual_pos=args.num_gradual_pos
        self.is_constrain=args.is_constrain 
        self.diagonosis = args.diag
        
    


    def read_pretrain_embedding(self) -> Tuple[Union[Dict[str, np.array], None], int]:
        """
        Read the pretrained word embeddings, return the complete embeddings and the embedding dimension
        """
        print("reading the pretraing embedding: %s" % (self.embedding_file))
        if self.embedding_file is None:
            print("pretrain embedding in None, using random embedding")
            return None, self.embedding_dim
        else:
            exists = os.path.isfile(self.embedding_file)
            if not exists:
                print(colored("[Warning] pretrain embedding file not exists, using random embedding",  'red'))
                return None, self.embedding_dim
                
        embedding_dim = -1
        embedding = dict()
        remove=[]
        with open(self.embedding_file, 'r', encoding='utf-8') as file:
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
        return embedding, embedding_dim

    

    def build_emb_table(self) -> None:
        """
        build the embedding table with pretrained word embeddings (if given otherwise, use random embeddings)
        :return:
        """
        count=0
        print("Building the embedding table for vocabulary...")
        scale = np.sqrt(3.0 / self.embedding_dim)
        if self.embedding is not None:
            print("[Info] Use the pretrained word embedding to initialize: %d x %d" % (len(self.word2idx), self.embedding_dim))
            self.word_embedding = np.empty([len(self.word2idx), self.embedding_dim])
            for word in self.word2idx:
                if word in self.embedding:
                    self.word_embedding[self.word2idx[word], :] = self.embedding[word]
                elif word.lower() in self.embedding:
                    self.word_embedding[self.word2idx[word], :] = self.embedding[word.lower()]
                else:
                    count+=1
                    self.word_embedding[self.word2idx[word], :] = np.random.uniform(-scale, scale, [1, self.embedding_dim])
            self.embedding = None
        else:
            self.word_embedding = np.empty([len(self.word2idx), self.embedding_dim])
            for word in self.word2idx:
                self.word_embedding[self.word2idx[word], :] = np.random.uniform(-scale, scale, [1, self.embedding_dim])
    