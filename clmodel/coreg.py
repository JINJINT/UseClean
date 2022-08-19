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
import torch.nn.functional as F


def kl_div(p, q):
    return (p * ((p + 1e-5).log() - (q + 1e-5).log())).sum(-1)

class onemodel(nn.Module):

    def __init__(self, config, print_info: bool = True):
        """
        config: a class that contains all the configurations
        print_info: whether print some information or not
        """
        super(onemodel, self).__init__()
        self.device = config.device
        
        self.label2idx = config.label2idx
        self.idx2label=config.idx2label

        self.Oid = self.label2idx['O']
        self.padid = self.label2idx['<PAD>']
        self.startid=self.label2idx['<START>']
        self.stopid=self.label2idx['<STOP>']

        self.contrastive = config.contrastive
        self.classifier = config.classifier
        config.random = True
        
        # model architecture
        if config.encoder=='bilstm':
            self.encoder = BiLSTMEncoder(config, print_info=print_info)
            self.pos_dic, self.type_dic = gen_dic(config.label2idx.keys(),self.label2idx)
            self.tags_num = len(self.idx2label)
            e_type,pos = gen_embedding_table(self.idx2label,self.type_dic,self.pos_dic)
            self.type_embedding = torch.nn.Embedding(self.tags_num, self.tags_num).from_pretrained(e_type,freeze=True).cuda(self.device)
            self.pos_embedding = torch.nn.Embedding(self.tags_num, self.tags_num).from_pretrained(pos,freeze=True).cuda(self.device)
        else: 
            self.encoder = TransEncoder(config, print_info=print_info)   
            self.word_embedding = self.encoder.word_embedding
            self.type_embedding = self.encoder.type_embedding
            self.pos_embedding = self.encoder.pos_embedding

        if config.classifier =='crf':
            self.inferencer = LinearCRF(config, print_info=print_info,reduce = True)
        else: # use mlp
            self.inferencer = mlp(config, print_info=print_info)     
                 
       
    @overrides
    def forward(self, batched_data) -> torch.Tensor:
        """
        Calculate the negative loglikelihood.
        param 
            batched_data: a dictionary contains all the needed info
        return 
            the loss with shape (1)
            the encoder_scores (i.e. logits) with shape (batch_size x max_seq_len x label_size)
        """

        batch_size = batched_data["input_ids"].size(0)
        sent_len = batched_data["input_ids"].size(1)
        word_len = max(batched_data["word_seq_lens"])
        
        # mask the padding position
        maskTemp = torch.arange(1, word_len + 1, dtype=torch.long).view(1, word_len).expand(batch_size, word_len).to(self.device)
        mask = torch.le(maskTemp, batched_data["word_seq_lens"].view(batch_size, 1).expand(batch_size, word_len)).to(self.device).float()
        
        # get the network score
        encoder_scores, contrastive_loss = self.encoder(batched_data["input_ids"], batched_data["token_seq_lens"], batched_data["attention_mask"], batched_data['token2word_mask'])
        
        if self.classifier =='crf':
            partition, score = self.inferencer(encoder_scores, batched_data["word_seq_lens"], batched_data["labels_id"], mask.bool())
            loss = (partition - score)/batch_size
        else: # use mlp
            loss = self.inferencer(encoder_scores, batched_data["word_seq_lens"], batched_data["labels_id"], mask.bool())
        return loss, encoder_scores # encoder_scores is just logits


class NNCRF_coreg(nn.Module):

    def __init__(self, config):
        super(NNCRF_coreg, self).__init__()
        self.device = config.device
        
        self.label2idx = config.label2idx
        self.idx2label=config.idx2label

        self.Oid = self.label2idx['O']
        self.padid = self.label2idx['<PAD>']
        self.startid=self.label2idx['<START>']
        self.stopid=self.label2idx['<STOP>']

        self.contrastive = config.contrastive
        
        # model architecture
        self.models = []
        for i in range(config.model_nums):
            if config.encoder=='bilstm':
                self.models.append(onemodel(config))
                if i==0: # use the first model for inference
                    self.pos_dic, self.type_dic = gen_dic(config.label2idx.keys(),self.label2idx)
                    self.tags_num = len(self.idx2label)
                    e_type,pos = gen_embedding_table(self.idx2label,self.type_dic,self.pos_dic)
                    self.type_embedding = torch.nn.Embedding(self.tags_num, self.tags_num).from_pretrained(e_type,freeze=True).cuda(self.device)
                    self.pos_embedding = torch.nn.Embedding(self.tags_num, self.tags_num).from_pretrained(pos,freeze=True).cuda(self.device)
            else: 
                self.models.append(onemodel(config))
                if i==0: # use the first model for inference
                    self.word_embedding = self.models[0].encoder.word_embedding
                    self.type_embedding = self.models[0].encoder.type_embedding
                    self.pos_embedding = self.models[0].encoder.pos_embedding
        
        self.encoder = self.models[0].encoder
        self.inferencer = self.models[0].inferencer
                 
    @overrides
    def forward(self, batched_data, alpha_t =0) -> torch.Tensor:
        """
        param 
            batched_data: (batch_size x max_seq_len)
            alpha_t: the coeficient before the regularization term
        return 
            the loss with shape (1)
        """

        batch_size = batched_data["input_ids"].size(0)
        sent_len = batched_data["input_ids"].size(1)
        word_len = max(batched_data["word_seq_lens"])

        num_models = len(self.models)
        outputs = []
        for i in range(num_models):
            output = self.models[i](batched_data)
            outputs.append(output)
        loss = sum([output[0] for output in outputs]) / num_models
        
        logits = [output[1] for output in outputs]
        probs = [F.softmax(logit, dim=-1) for logit in logits]
        avg_prob = torch.stack(probs, dim=0).mean(0)  
        
        # mask the padding position
        if alpha_t>0:
            maskTemp = torch.arange(1, word_len + 1, dtype=torch.long).view(1, word_len).expand(batch_size, word_len).to(self.device)
            mask = torch.le(maskTemp, batched_data["word_seq_lens"].view(batch_size, 1).expand(batch_size, word_len)).to(self.device).float()
            reg_loss = sum([kl_div(avg_prob, prob) * mask for prob in probs]) / num_models
            reg_loss = reg_loss.sum() / (mask.sum() + 1e-3)
        else:
            reg_loss = 0   
        # print('reg_loss: %.5f'%(reg_loss)) # ~0.02
        loss = loss + alpha_t * reg_loss
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

