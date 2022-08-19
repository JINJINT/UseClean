
from distutils.command.config import config
import torch
import torch.nn as nn
from utils.trans_utils import context_models
import numpy as np
# from datastruct import ContextEmb
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from overrides import overrides

from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertConfig, BertEncoder, BertEmbeddings

from typing import List, Optional, Tuple

from dataclasses import dataclass
from transformers.file_utils import (
    ModelOutput
)

class TransEncoder(nn.Module):

    def __init__(self, config, print_info: bool = True):
        """
        param
            config: a class that contains all the configurations
            print_info: whether print some information or not
        """
        super(TransEncoder, self).__init__()
        output_hidden_states = False ## to use all hidden states or not
        
        self.label_size = config.label_size
        self.device = config.device
        self.label2idx = config.label2idx
        self.labels = config.idx2label
        self.contrastive = config.contrastive
        self.tau = config.tau
        self.ent_freq = config.ent_freq
        
        print(f"[Model Info] Loading pretrained language model {config.encoder}")

        self.model = context_models[config.encoder]["model"].from_pretrained(config.encoder,
                                                                             output_hidden_states= output_hidden_states,
                                                                             return_dict=False).to(self.device)
        if config.random:
            self.model.init_weights()

        config.hidden_dim = self.model.config.hidden_size

        if self.contrastive:
            self.mlp = MLPLayer(config).to(self.device)
            self.sim = Similarity(temp=0.05).to(self.device)
        
        self.hidden2tag = nn.Linear(self.model.config.hidden_size, self.label_size).to(self.device)
        self.sigmoid = nn.Sigmoid().to(self.device)
        
        self.word_embedding = self.model.embeddings.word_embeddings.to(self.device)
        self.type_embedding = self.model.embeddings.token_type_embeddings.to(self.device)
        self.pos_embedding = self.model.embeddings.position_embeddings.to(self.device)

        if print_info:
            #print("[Model Info] Input size to encoder: {}".format(config.hidden_dim))
            print("[Model Info] encoder Final hidden size: {}".format(config.hidden_dim))


    @overrides
    def forward(self, input_ids: torch.Tensor,
                      word_seq_tensor: torch.Tensor,
                      attention_mask: torch.Tensor,
                      token2word_ids: torch.Tensor) -> torch.Tensor:
        """
        Encoding the input with bert
        param 
           input_ids: (batch_size, sent_token_len), the ids of each token in the sentence
           word_seq_lens: (batch_size, 1)
           attention_mask: (batch_size, sent_token_len), denote which positions are paddings 
           token2word_ids: (batch_size, sent_len), the ids of the first token for each word
        return 
           outputs: (batch_size, sent_len, label_size) the logits matrix 
           contrastive loss: the contrastive loss, will be zero if not using contrastive learning
        """
        # if using contrastive learning SimCSE, one should only use this when using bert encoder
        if self.contrastive:
            # create two copies of the batch
            input_ids_aug = torch.zeros((input_ids.shape[0]*2,input_ids.shape[1]),dtype=torch.long)
        
            for i in range(input_ids.shape[0]):
                input_ids_aug[2*i,:] = input_ids[i,:]
                input_ids_aug[2*i+1,:] = input_ids[i,:]
            input_ids_aug = input_ids_aug.to(self.device)
            
            # also copy the attention mask
            attention_mask_aug = torch.zeros((attention_mask.shape[0]*2,attention_mask.shape[1]),dtype=torch.long)
            for i in range(attention_mask.shape[0]):
                attention_mask_aug[2*i,:] = attention_mask[i,:]
                attention_mask_aug[2*i+1,:] = attention_mask[i,:]    
            attention_mask_aug = attention_mask_aug.to(self.device)
            
            # put the data into the bert encoder
            word_rep,_ = self.model(input_ids_aug, attention_mask_aug)
            # sentence embedding, using CLS token
            pooler_output = word_rep[:,0] 
            pooler_output = pooler_output.view((input_ids.shape[0], 2, pooler_output.size(-1)))
            pooler_output = self.mlp(pooler_output)
            # get the embeddings for two different views
            z1, z2 = pooler_output[:,0], pooler_output[:,1]
            # compute their cosine similarity
            cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
            labels = torch.arange(cos_sim.size(1)).long().to(self.device)
            # compute the contrastive loss
            loss_fct = nn.CrossEntropyLoss()
            contrastive_loss = loss_fct(cos_sim, labels)
            # get the word hidden represenations 
            word_rep = word_rep[[2*i for i in range(input_ids.shape[0])],:,:]
        else:
            word_rep,_ = self.model(input_ids, attention_mask)
            contrastive_loss = 0            
        
        batch_size, _, rep_size = word_rep.size()
        _, max_sent_len = token2word_ids.size()
        final_word_rep = torch.gather(word_rep[:, :, :], 1, token2word_ids.unsqueeze(-1).expand(batch_size, max_sent_len, rep_size))
        outputs = self.hidden2tag(final_word_rep)

        # balance: adjust the logits by entity class frequency
        if self.tau>0:
            outputs = outputs + self.tau * ((torch.tensor(self.ent_freq, dtype=torch.long) + 1e-12).log()).to(self.device)
        
        return outputs, contrastive_loss   


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """
    

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

        
class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        else:
            raise NotImplementedError

class Arguments():
    def __init__(self): 
        self.temp = 0.05 # Temperature for softmax.
        self.pooler_type = 'cls' #What kind of pooler to use 
        # Number of sentences in one instance
        # 2: pair instance; 3: pair instance with a hard negative
        self.num_sent = 2