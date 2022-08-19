from copy import deepcopy
import torch
import torch.nn as nn
from config import START, STOP, PAD
from utils import log_sum_exp_pytorch, gen_dic, gen_embedding_table
from evaluation import evaluate_conf
from encoder.charbilstm import CharBiLSTM
from encoder.bilstm_encoder import BiLSTMEncoder
from encoder.trans_encoder import TransEncoder
from nerlayer.linear_crf_inferencer import LinearCRF
from nerlayer.mlp_inferencer import mlp
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# from datastruct import ContextEmb
from typing import Tuple
from overrides import overrides
import numpy as np


class NNCRF_baseline(nn.Module):

    def __init__(self, config, model_init = None, print_info: bool = True, reduce = True):
        """
        param
            config: a class that contains all the configurations
            model_init: a full model for warm start (not gonna be used here in baseline)
            print_info: whether print some information or not
            reduce: in forward function, where outout aggregated resuts over batch or single result for each sample in the batch
        """
        super(NNCRF_baseline, self).__init__()
        self.device = config.device
        
        self.label2idx = config.label2idx
        self.idx2label=config.idx2label

        self.Oid = self.label2idx['O']
        self.padid = self.label2idx['<PAD>']
        self.startid=self.label2idx['<START>']
        self.stopid=self.label2idx['<STOP>']

        self.contrastive = config.contrastive # whether add contrastive learning regularization
        self.lamb = config.lamb # the coefficient of the contrastive learning regularization term
        self.classifier = config.classifier
        self.reduce = reduce
        
        # model architecture
        if config.encoder=='bilstm':
            self.encoder = BiLSTMEncoder(config, print_info=print_info)   
        else: 
            self.encoder = TransEncoder(config, print_info=print_info)     
            self.word_embedding = self.encoder.word_embedding
        
        if config.classifier =='crf':
            self.inferencer = LinearCRF(config, print_info=print_info, reduce = self.reduce)
        else: # use mlp
            self.inferencer = mlp(config, print_info=print_info)       
  
       
    @overrides
    def forward(self, batched_data, train = True) -> torch.Tensor:
        """
        Calculate the negative loglikelihood.
        param
            batched_data: a dictionary contains all the needed data
        return
            the loss with shape (batch_size) if reduce=false or with shape (1) of reduce=true
        """

        batch_size = batched_data["input_ids"].size(0)
        sent_len = batched_data["input_ids"].size(1)
        word_len = max(batched_data["word_seq_lens"])
        
        # mask the padding position
        maskTemp = torch.arange(1, word_len + 1, dtype=torch.long).view(1, word_len).expand(batch_size, word_len).to(self.device)
        mask = torch.le(maskTemp, batched_data["word_seq_lens"].view(batch_size, 1).expand(batch_size, word_len)).to(self.device).float()
        
        # get the network score
        encoder_scores, contrastive_loss = self.encoder(batched_data["input_ids"], batched_data["token_seq_lens"], batched_data["attention_mask"], batched_data['token2word_mask'])
        if self.classifier=='crf':
            partition, score = self.inferencer(encoder_scores, batched_data["word_seq_lens"], batched_data["labels_id"], mask.bool())
            # if not train:
            #     print('contrastive: %.5f'%(contrastive_loss))
            if self.contrastive:
                return partition - score + self.lamb * contrastive_loss
            else:
                return partition - score
        else: # use mlp
            loss = self.inferencer(encoder_scores, batched_data["word_seq_lens"], batched_data["labels_id"], mask.bool())
            return loss
        
 
    def decode(self, batched_data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode the batch input
        param 
            batched_data: the batched data as a dictionary
        return
            bestScores: the best utterance score
            decodeIdx: the corresponding utterance tag id sequence
        """
        
        features,_ = self.encoder(batched_data['input_ids'], batched_data["token_seq_lens"], batched_data['attention_mask'], batched_data['token2word_mask'])
        bestScores, decodeIdx = self.inferencer.decode(features, batched_data['word_seq_lens'])
        
        return bestScores, decodeIdx



