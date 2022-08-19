from distutils.command.config import config
from pickle import NONE
import torch
import torch.nn as nn
from config import START, STOP, PAD
from utils import log_sum_exp_pytorch, gen_dic, gen_embedding_table
from evaluation import evaluate_conf, evaluate_confscore
from encoder.charbilstm import CharBiLSTM
from encoder.bilstm_encoder import BiLSTMEncoder
from encoder.trans_encoder import TransEncoder
from nerlayer.linear_partial_crf_inferencer import LinearCRFpartial
from nerlayer.linear_crf_inferencer import LinearCRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# from datastruct import ContextEmb
from typing import Tuple
from overrides import overrides
import numpy as np
from torch.distributions import Categorical
from nerlayer.mlp_inferencer import mlp
from copy import deepcopy

class NNCRF_sl(nn.Module):

    def __init__(self, config, print_info: bool = True, model_init= None):
        """
        config: a class that contains all the configurations
        model_init: a full model for warm start (not gonna be used here in baseline)
        print_info: whether print some information or not
        """
        super(NNCRF_sl, self).__init__()
        self.device = config.device
        self.score = config.score
        self.usescore = config.usescore
        self.cutoff = config.cutoff
        self.diagonosis = config.diagonosis
        self.contrastive = config.contrastive
        self.lamb = config.lamb
        self.weight = config.weight

        # model architecture
        if config.encoder=='bilstm':
            if model_init is None:
                self.encoder = BiLSTMEncoder(config, print_info=print_info)
            else:
                self.encoder = deepcopy(model_init.encoder)    
        else: 
            if model_init is None:
                self.encoder = TransEncoder(config, print_info=print_info)
            else:
                self.encoder = deepcopy(model_init.encoder)      
            self.word_embedding = self.encoder.word_embedding
            
        if config.classifier =='crf':
            self.inferencer = LinearCRFpartial(config, print_info=print_info)
            if model_init is not None:
                self.inferencer.transition = deepcopy(model_init.inferencer.transition)
        else: # use mlp
            if model_init is None:
                self.inferencer = mlp(config, print_info=print_info) 
            else:
                self.inferencer = model_init.inferencer   
                 
        self.label2idx = config.label2idx
        self.idx2label =config.idx2label
        
        self.Oid = self.label2idx['O']
        self.padid = self.label2idx['<PAD>']
        self.startid=self.label2idx['<START>']
        self.stopid=self.label2idx['<STOP>']
        
        self.pos_dic, self.type_dic = gen_dic(config.label2idx.keys(),self.label2idx)
        
        self.tags_num = len(self.idx2label)
        e_type, pos = gen_embedding_table(self.idx2label,self.type_dic,self.pos_dic)
        self.type_embedding = torch.nn.Embedding(self.tags_num, self.tags_num).from_pretrained(e_type,freeze=True).cuda(self.device)
        self.pos_embedding=torch.nn.Embedding(self.tags_num, self.tags_num).from_pretrained(pos,freeze=True).cuda(self.device)
        
    @overrides
    def forward(self, batched_data, 
                    forget_rate_neg=0,forget_rate_pos=0,is_constrain=False,
                    train = True,
                    compute_conf = True,
                    use_label_tag_mask = True, 
                    label_tag_mask = None
                    ) -> torch.Tensor:
        """
        Calculate the negative loglikelihood.
        param 
            batched_data: a dictionary contains all the need data info
            forget_rate_neg, forget_rate_pos: the remove rate for nonentity and entity tokens
            is_constrain: whether use further calibration based on entity position and types
            train: whether we are in the training mode (this is used for computing some statistics that only training phase can have: i.e. noisy versus clean F1)
            compute_conf: whether compute confidece score related evaluation (this argument is automatically assigned based on your model and cutoff choices, so leave it be)
            use_label_tag_mask: whether to use precomputed label_tag_mask
            label_tag_mask: (batch_size, sent_len) which word we should trust
        return 
            the loss with shape (1);
            some additional ones if in the train mode:
                conf_results: how well the sample selection agrees with the true clean/noisy label
                avg_confscore_pos, avg_confscore_neg:  the averaged confidence score for entity and nonentity
                confscore: (batch_size, sent_len) the confidence score matrix (larger=less_confident)
        """

        batch_size = batched_data['input_ids'].size(0)
        sent_len = max(batched_data['word_seq_lens'])
        
        # mask the padding position
        maskTemp = torch.arange(1, sent_len + 1, dtype=torch.long).view(1, sent_len).expand(batch_size, sent_len).to(self.device)
        mask = torch.le(maskTemp, batched_data['word_seq_lens'].view(batch_size, 1).expand(batch_size, sent_len)).to(self.device).float()
        
        # get the network score
        encoder_scores, contrastive_loss = self.encoder(batched_data['input_ids'], batched_data['word_seq_lens'], batched_data['attention_mask'], batched_data['token2word_mask'])
        # print(encoder_scores.shape) #(batchsize, max_seqlen, label_size)

        onehot_label = torch.zeros_like(encoder_scores).scatter_(-1, batched_data['labels_id'].unsqueeze(-1), 1)
        
        # wether the tag is nonentity or not
        negative_mask=torch.eq(batched_data['labels_id'], self.Oid).float()* mask.float()

        # whether the tag is entity or not
        positive_mask =torch.ne(batched_data['labels_id'], self.Oid).float()* mask.float()

        confscore = None

        if train: 
            #========== what kind of measure to use as confidence score
            if self.score=="encoderloss":       
                #==== use the LSTM loss to determine the confidence score
                token_prob = torch.nn.functional.log_softmax(encoder_scores, dim=-1)
                forward_loss = -(onehot_label * token_prob).sum(dim=-1) * mask
                confscore=forward_loss.detach()

            if self.score=='nerloss':
                #==== use the NER loss to determine the confidence score
                log_marginals = self.inferencer.marginal(encoder_scores, batched_data['word_seq_lens'])
                token_prob = log_marginals
                forward_loss = -(onehot_label * token_prob).sum(dim=-1) * mask
                confscore = forward_loss.detach()

            if self.score=='diff': 
                #==== use the LSTM loss before softmax to determine the confidence score
                # mask out target values
                maxscore = torch.max(encoder_scores, dim=-1).values
                obsscore = (encoder_scores * onehot_label).sum(dim=-1)
                forward_loss = (maxscore-obsscore) * mask
                confscore = forward_loss.detach()

            if self.score=='aum':
                #==== use the LSTM loss before softmax to determine the confidence score
                # mask out target values
                masked_logits = encoder_scores + onehot_label*(-100000.0)
                maxscore = torch.max(masked_logits, dim=-1).values
                obsscore = (encoder_scores * onehot_label).sum(dim=-1)
                forward_loss = (obsscore-maxscore) * mask
                confscore = forward_loss.detach()    

            if self.score=='entropy':
                token_prob = torch.nn.functional.softmax(encoder_scores, dim=-1)
                token_entropy = Categorical(token_prob).entropy()
                forward_loss = token_entropy * mask
                confscore=forward_loss.detach()

            if self.score=='spike':
                token_prob = torch.nn.functional.softmax(encoder_scores, dim=-1)
                maxscore = torch.max(token_prob, dim=-1).values
                forward_loss = maxscore * mask
                confscore=forward_loss.detach()    

            if self.score=='cross':
                #==== use cross-validation score
                forward_loss = batched_data['scores'] * mask
                confscore = forward_loss.detach()

            if self.score=='data':   
                #==== use data-based score
                forward_loss = batched_data['scores'] * mask
                confscore = forward_loss.detach() 

            if self.score in ['useclean','usecleantail','usecleanhead','usecontrast']:   
                #==== use clean data predicted score
                forward_loss = batched_data['scores'] * mask
                confscore = forward_loss.detach()     

            avg_confscore_neg=(forward_loss*negative_mask).sum()/(negative_mask.sum()+1e-6)
            avg_confscore_pos=(forward_loss*positive_mask).sum()/(positive_mask.sum()+1e-6)    
           
            #========== what to do with confidence score
            conf_results = {}
            #=== toss
            if self.usescore == 'toss':  
                if not use_label_tag_mask: # do not discard any words
                    label_tag_mask = torch.ones_like(batched_data['labels_id']).float()*mask.float()
                    label_tag_mask =  label_tag_mask.detach()
                    if train and compute_conf:
                        conf_results=evaluate_conf(batched_data['gold_labels_id'], batched_data['labels_id'], label_tag_mask, mask, negative_mask)
                    partial_label=torch.ones_like(onehot_label)
                    label_tag_mask = label_tag_mask.unsqueeze(-1)*onehot_label + (1-label_tag_mask).unsqueeze(-1)*partial_label   
                else:
                    if self.cutoff == 'heuri':
                        #====== get which nonconfident word-nonentity to toss
                        # get the loss from the encoder model, with padding position adjusted
                        tmp = confscore.view(batch_size * sent_len) + (1000 * (1 - mask).view(batch_size * sent_len)) + (
                                        1000 * (positive_mask.view(batch_size * sent_len)))

                        # sort the instances by their loss value
                        index=torch.argsort(tmp, dim=-1) # increasing order
                        
                        remember_rate_neg = 1.0 - forget_rate_neg
                        num_remember = int(remember_rate_neg * (negative_mask.sum()))
                        small_loss_index = index[:num_remember] # toss the instance with large loss value
                        
                        # mask whether the instance should be tossed or not
                        small_loss_mask_neg = torch.zeros_like(tmp)
                        for num in small_loss_index:
                            small_loss_mask_neg[num] = 1 # keep only instances with small loss
                        small_loss_mask_neg = small_loss_mask_neg.view((batch_size, sent_len))
                        
                        if num_remember == 0:
                            small_loss_mask_neg = negative_mask
                        
                        
                        #====== get which nonconfident word-entity to toss

                        tmp = confscore.view(batch_size * sent_len) + (1000 * (1 - mask).view(batch_size * sent_len)) + (
                                    1000 * (negative_mask.view(batch_size * sent_len)))

                        index=torch.argsort(tmp, dim=-1)
                        remember_rate_pos = 1.0 - forget_rate_pos
                        
                        num_remember = int(remember_rate_pos * (positive_mask.sum()))
                        small_loss_index = index[:num_remember]
                        small_loss_mask_pos = torch.zeros_like(tmp)
                        for num in small_loss_index:
                            small_loss_mask_pos[num] = 1
                        small_loss_mask_pos = small_loss_mask_pos.view((batch_size, sent_len))
                        if num_remember == 0:
                            small_loss_mask_pos = positive_mask
                        
                        #===== get which word-tag to toss in all
                        small_loss_mask = (small_loss_mask_pos.bool() + small_loss_mask_neg.bool()).float()
                        small_loss_mask = small_loss_mask.detach()

                        # evaluate the current tossing
                        if train and compute_conf: 
                            conf_results=evaluate_conf(batched_data['gold_labels_id'], batched_data['labels_id'], small_loss_mask, mask, negative_mask)    

                        #===== calibrate further which instance to toss for the pos-type kind of entity
                        partial_label=torch.ones_like(onehot_label)
                        
                        type_lookup = self.type_embedding(batched_data['labels_id'])
                        pos_lookup=self.pos_embedding(batched_data['labels_id'])
                        
                        prob = torch.nn.functional.softmax(encoder_scores, dim=-1)
                        prob=prob.detach()

                        type_prob=(prob*type_lookup).mean(dim=-1)
                        pos_prob=(prob*pos_lookup).mean(dim=-1)
                        type_change_mask=(type_prob>pos_prob)*mask*(1-small_loss_mask)
                        pos_change_mask=(type_prob<pos_prob)*mask*(1-small_loss_mask)
                        change_label=((type_change_mask.unsqueeze(-1)*type_lookup)+(pos_change_mask.unsqueeze(-1)*pos_lookup))+((1-small_loss_mask)*(1-type_change_mask)*(1-pos_change_mask)).unsqueeze(-1)*partial_label
                        
                        #===== the final mask, which defines which word-tag to toss all together
                        if(is_constrain):
                            label_tag_mask=small_loss_mask.unsqueeze(-1)*onehot_label + (1-small_loss_mask).unsqueeze(-1)*change_label
                        else:
                            label_tag_mask = small_loss_mask.unsqueeze(-1)*onehot_label + (1-small_loss_mask).unsqueeze(-1)*partial_label
                        
                    if self.cutoff == 'oracle':
                        scorevalue = confscore.view(batch_size * sent_len).clone()
                        scorevalue = scorevalue.tolist() 
                        score_results, oracle_label_tag_mask = evaluate_confscore(batched_data['gold_labels_id'], scorevalue, batched_data['labels_id'], mask, negative_mask, batch_size, batched_data['word_seq_lens'], self.device)   

                        label_tag_mask = oracle_label_tag_mask.float().detach()
                        if train and compute_conf:
                            conf_results=evaluate_conf(batched_data['gold_labels_id'], batched_data['labels_id'], label_tag_mask, mask, negative_mask)
                        partial_label=torch.ones_like(onehot_label)
                        label_tag_mask = label_tag_mask.unsqueeze(-1)*onehot_label + (1-label_tag_mask).unsqueeze(-1)*partial_label

                    if self.cutoff in ['goracle','fake','fitmix']:
                        # find the cutoff via regressing the confsocre on true_is_noisy response per epoch
                        label_tag_mask = label_tag_mask.float()*mask.float()
                        label_tag_mask =  label_tag_mask.detach()
                        if train and compute_conf:
                            conf_results=evaluate_conf(batched_data['gold_labels_id'], batched_data['labels_id'], label_tag_mask, mask, negative_mask)
                        partial_label=torch.ones_like(onehot_label)
                        label_tag_mask = label_tag_mask.unsqueeze(-1)*onehot_label + (1-label_tag_mask).unsqueeze(-1)*partial_label    

                    if self.cutoff == 'clean': 
                        label_tag_mask = torch.eq(batched_data['gold_labels_id'], batched_data['labels_id']).float()*mask.float()
                        label_tag_mask =  label_tag_mask.detach()
                        if train and compute_conf:
                            conf_results=evaluate_conf(batched_data['gold_labels_id'], batched_data['labels_id'], label_tag_mask, mask, negative_mask)
                        partial_label=torch.ones_like(onehot_label)
                        label_tag_mask = label_tag_mask.unsqueeze(-1)*onehot_label + (1-label_tag_mask).unsqueeze(-1)*partial_label    
                     
                label_tag_mask=label_tag_mask.detach()
                        

            #=== compute the masked loss
            if self.weight:
                unlabled_score, labeled_score = self.inferencer(encoder_scores, batched_data['word_seq_lens'], label_tag_mask, batched_data['weights'])
            else:
                unlabled_score, labeled_score = self.inferencer(encoder_scores, batched_data['word_seq_lens'], label_tag_mask)

            
            if train and self.cutoff == 'oracle':
                for scorename in score_results.keys():
                    conf_results[scorename] = score_results[scorename]    

            return unlabled_score - labeled_score + self.lamb * contrastive_loss, conf_results, avg_confscore_pos, avg_confscore_neg, confscore
        
        else: # if not train
            unlabled_score, labeled_score = self.inferencer(encoder_scores, batched_data['word_seq_lens'], None)
            return unlabled_score - labeled_score + self.lamb * contrastive_loss

 
    def decode(self, batched_data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode the batch input
        param 
            batched_data: the batched data as a dictionary
        return
            bestScores: the best utterance score
            decodeIdx: the corresponding utterance tag id sequence
        """
        
        features,_ = self.encoder(batched_data["input_ids"], batched_data["word_seq_lens"], batched_data['attention_mask'], batched_data['token2word_mask'])
        bestScores, decodeIdx = self.inferencer.decode(features, batched_data["word_seq_lens"])
        
        return bestScores, decodeIdx



