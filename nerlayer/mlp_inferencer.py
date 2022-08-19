from json import encoder
import torch.nn as nn
import torch
import numpy as np
from utils import log_sum_exp_pytorch
from config import START, STOP, PAD
from typing import Tuple
from overrides import overrides
from utils import torch_model_utils as tmu 

class mlp(nn.Module):


    def __init__(self, config):
        super(mlp, self).__init__()

        self.label_size = config.label_size
        self.device = config.device
        self.use_char = config.use_char_rnn
        self.context_emb = config.context_emb
        
        self.label2idx = config.label2idx
        self.labels = config.idx2label
        self.start_idx = self.label2idx[START]
        self.end_idx = self.label2idx[STOP]
        self.pad_idx = self.label2idx[PAD]
        
        self.loss_fnt = nn.CrossEntropyLoss(ignore_index=-1)
        

    @overrides
    def forward(self, encoder_scores, word_seq_lens, tags, mask):
        """
        Calculate the loss
        param
            encoder_scores: (batch_size, sent_len, label_size) the logits matrix
            word_seq_lens: (batch_size, 1) the length of each utterance
            tags: (batch, sent_len) observed labels ids
            mask: this gonna not be used here, keep just for unified form
        return
            loss
        """
        logits = encoder_scores.view(-1, self.label_size)
        tags = tags.view(-1)
        loss = self.loss_fnt(logits, tags)
        return loss

    def decode(self, encoder_scores, wordSeqLengths, annotation_mask = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode the batch input
        param
            encoder_scores: (batch_size, sent_len, label_size) the logits matrix
            wordSeqLengths: this gonna not be used here, keep just for unified form
            attention_mask: this gonna not be used here, keep just for unified form
        return
            bestscores: will be None here, keep just for unified form
            bestids: (batch, sent_len) the predicted labels ids
        """
        logits = encoder_scores.view(-1, self.label_size)
        bestids = torch.argmax(logits, axis=-1)
        bestids = bestids.view(encoder_scores.size()[0], encoder_scores.size()[1])
        bestscores = None
        return bestscores, bestids

