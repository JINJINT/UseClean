import torch.nn as nn
import torch

from utils import torch_model_utils as tmu 
from utils import log_sum_exp_pytorch
from config import START, STOP, PAD
from typing import Tuple
from overrides import overrides

class LinearCRFpartial(nn.Module):
    def __init__(self, config, print_info: bool = True):
        """
        param
            config: a class that contains all the configurations
            print_info: whether print some information or not
        """
        super(LinearCRFpartial, self).__init__()

        self.label_size = config.label_size
        self.device = config.device

        self.label2idx = config.label2idx
        self.labels = config.idx2label
        self.start_idx = self.label2idx[START]
        self.end_idx = self.label2idx[STOP]
        self.pad_idx = self.label2idx[PAD]

        # initialize the following transition (anything never -> start. end never -> anything. Same thing for the padding label)
        init_transition = torch.randn(self.label_size, self.label_size).to(self.device)
        init_transition[:, self.start_idx] = -10000.0
        init_transition[self.end_idx, :] = -10000.0
        init_transition[:, self.pad_idx] = -10000.0
        init_transition[self.pad_idx, :] = -10000.0

        self.transition = nn.Parameter(init_transition)

    @overrides
    def forward(self, encoder_scores, word_seq_lens, label_tag_mask, weights=None):
        """
        Calculate the (marginalized) negative log-likelihood (with weights) 
        param
            encoder_scores: (batch_size, sent_len, label_size) the logits matrix
            word_seq_lens: (batch_size, 1) the length of each utterance
            label_tag_mask: (batch, sent_len, label_size) 0/1 matrix, 
                            (k,i,j) indicating whether the word_i-tag_j pair is possible in the k-th utterance
            weights: (batch, sent_len, label_size) float matrix, indicating the weights for each word-tag pair;
                     if None, then we consider uniform weights for each word-tag pair
        return
            unlabeled_score: the partition function
            labeled_score: the numerator
        """
        all_scores = self.calculate_all_scores(encoder_scores= encoder_scores)
        unlabed_score = self.forward_unlabeled(all_scores, word_seq_lens)
        labeled_score = self.forward_labeled(all_scores, word_seq_lens, label_tag_mask) if weights is None else \
                        self.forward_labeled_with_marginal(all_scores, word_seq_lens, label_tag_mask, weights)
        return unlabed_score , labeled_score    


    def forward_unlabeled(self, all_scores: torch.Tensor, word_seq_lens: torch.Tensor, return_alpha=False) -> torch.Tensor:
        """
        Calculate the scores with the forward algorithm. Basically calculating the normalization term
        param
            all_scores: (batch_size, sent_len, num_labels, num_labels) from (encoder scores + transition scores).
            word_seq_lens: 
            return_alpha: whether return both the forward variable and the partition function
        return
            alpha, zs: alpha of shape (batch_size, sent_len, label_size) is the forward variable, 
                       and zs of the shape (batch_size, label_size) is the partition function if return_alpha is true
            Z: of shape (1) the partition function if return_alpha is false
        """
        batch_size = all_scores.size(0)
        seq_len = all_scores.size(1)
        alpha = torch.zeros(batch_size, seq_len, self.label_size).to(self.device)

        alpha[:, 0, :] = all_scores[:, 0,  self.start_idx, :] ## the first position of all labels = (the transition from start - > all labels) + current emission.

        for word_idx in range(1, seq_len):
            ## batch_size, word_size, label_size
            before_log_sum_exp = alpha[:, word_idx-1, :].view(batch_size, self.label_size, 1).expand(batch_size, self.label_size, self.label_size) + all_scores[:, word_idx, :, :]
            alpha[:, word_idx, :] = log_sum_exp_pytorch(before_log_sum_exp)

        ### batch_size x label_size
        last_alpha = torch.gather(alpha, 1, word_seq_lens.view(batch_size, 1, 1).expand(batch_size, 1, self.label_size)-1).view(batch_size, self.label_size)
        last_alpha += self.transition[:, self.end_idx].view(1, self.label_size).expand(batch_size, self.label_size)
        last_alpha = log_sum_exp_pytorch(last_alpha.view(batch_size, self.label_size, 1)).view(batch_size)

        if(return_alpha): 
            return alpha, last_alpha
        else:
            return torch.sum(last_alpha)


    def forward_labeled(self, all_scores: torch.Tensor, word_seq_lens: torch.Tensor, label_tag_mask: torch.Tensor) -> torch.Tensor:
        """
        Calculate the scores for the compatible instances.
        :param all_scores: (batch, seq_len, label_size, label_size)
        :param word_seq_lens: (batch, seq_len)
        :param label_tag_mask: (batch, seq_len, label_size) # whether-toss-or-not mask
        :return: sum of score for the compatible sequences Shape: (batch_size)
        """
        batch_size = all_scores.size(0)
        seq_len = all_scores.size(1)
        
        label_tag_mask = label_tag_mask.float().log()
        ## alpha is a log-space score
        alpha = torch.zeros(batch_size, seq_len, self.label_size).to(self.device)

        alpha[:, 0, :] = all_scores[:, 0,  self.start_idx, :] ## the first position of all labels = (the transition from start - > all labels) + current emission.
        alpha[:, 0, :] += label_tag_mask[:, 0, :]

        for word_idx in range(1, seq_len):
            ## batch_size, self.word_size, self.label_size
            before_log_sum_exp = alpha[:, word_idx-1, :].view(batch_size, self.label_size, 1).expand(batch_size, self.label_size, self.label_size) + all_scores[:, word_idx, :, :]
            alpha[:, word_idx, :] = log_sum_exp_pytorch(before_log_sum_exp)
            alpha[:, word_idx, :] += label_tag_mask[:, word_idx,:]

        ### batch_size x label_size
        last_alpha = torch.gather(alpha, 1, word_seq_lens.view(batch_size, 1, 1).expand(batch_size, 1, self.label_size)-1).view(batch_size, self.label_size)
        last_alpha += self.transition[:, self.end_idx].view(1, self.label_size).expand(batch_size, self.label_size)
        last_alpha = log_sum_exp_pytorch(last_alpha.view(batch_size, self.label_size, 1)).view(batch_size)

        return torch.sum(last_alpha)  

    def forward_labeled_with_marginal(self, all_scores: torch.Tensor, word_seq_lens: torch.Tensor, label_tag_mask: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Calculate the scores with the forward algorithm. Basically calculating the normalization term
        :param all_scores: (batch_size x max_seq_len x num_labels) from lstm scores.
        :param word_seq_lens: (batch_size)
        :param weights: shape (batch x max_seq_len x num_labels), the log-space score of the marginal probability
        :return: (batch_size) for the normalization scores
        """
        batch_size = all_scores.size(0)
        seq_len = all_scores.size(1)
        alpha = torch.zeros(batch_size, seq_len, self.label_size).to(self.device)
        label_tag_mask = label_tag_mask.float().log()

        alpha[:, 0, :] = all_scores[:, 0, self.start_idx, :]  ## the first position of all labels = (the transition from start - > all labels) + current emission.
        alpha[:, 0, :] += label_tag_mask[:, 0,:]
        alpha[:, 0, :] += weights[:, 0, :]

        for word_idx in range(1, seq_len):
            ## batch_size, self.label_size, self.label_size
            before_log_sum_exp = alpha[:, word_idx - 1, :].view(batch_size, self.label_size, 1).expand(batch_size, self.label_size, self.label_size) \
                                 + all_scores[:, word_idx, :, :]
            alpha[:, word_idx, :] = log_sum_exp_pytorch(before_log_sum_exp)
            alpha[:, word_idx, :] += label_tag_mask[:, word_idx,:]
            alpha[:, word_idx, :] += weights[:, word_idx, :]

        ### batch_size x label_size
        last_alpha = torch.gather(alpha, 1,  word_seq_lens.view(batch_size, 1, 1).
                                  expand(batch_size, 1, self.label_size) - 1).view(batch_size, self.label_size)
        last_alpha += self.transition[:, self.end_idx].view(1, self.label_size).expand(batch_size, self.label_size)
        last_alpha = log_sum_exp_pytorch(last_alpha.view(batch_size, self.label_size, 1)).view(batch_size)

        return torch.sum(last_alpha)    


    def calculate_all_scores(self, encoder_scores: torch.Tensor) -> torch.Tensor:
        """
        Calculate all scores by adding up the transition scores and emissions (i.e. encoder score from encoder).
        Basically, compute the scores for each edges between labels at adjacent positions.
        This score is later be used for forward-backward inference
        param 
            encoder_scores: (batch_size, sent_len, label_size) emission scores.
        return
            scores: (batch_size, sent_len, label_size, label_size) scores for each edges between labels at adjacent positions
        """
        batch_size = encoder_scores.size(0)
        seq_len = encoder_scores.size(1)
        scores = self.transition.view(1, 1, self.label_size, self.label_size).expand(batch_size, seq_len, self.label_size, self.label_size) + \
                 encoder_scores.view(batch_size, seq_len, 1, self.label_size).expand(batch_size, seq_len, self.label_size, self.label_size)
        return scores

    def decode(self, features, wordSeqLengths) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode the batch input
        param 
            feaures: (batch_size, sent_len, label_size) emission scores
            wordSeqLengths: (batch_size,) the length of words for each utterance
        return
            the best scores as well as the predicted label ids.
                (batch_size) and (batch_size x max_seq_len) 
        """
        all_scores = self.calculate_all_scores(features)
        bestScores, decodeIdx = self.constrainted_viterbi_decode(all_scores, wordSeqLengths)
        return bestScores, decodeIdx

    def constrainted_viterbi_decode(self, all_scores: torch.Tensor, word_seq_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Use viterbi to decode the instances given the scores and transition parameters
        param 
            all_scores: (batch_size, sent_len , label_size, label_size)
            word_seq_lens: (batch_size,) the length of words for each utterance
        return
            the best scores as well as the predicted label ids.
               (batch_size,) and (batch_size , sent_len)
        """
        
        batchSize = all_scores.shape[0]
        sentLength = all_scores.shape[1]
        
        scoresRecord = torch.zeros([batchSize, sentLength, self.label_size]).to(self.device)
        idxRecord = torch.zeros([batchSize, sentLength, self.label_size], dtype=torch.int64).to(self.device)
        mask = torch.ones_like(word_seq_lens, dtype=torch.int64).to(self.device)
        startIds = torch.full((batchSize, self.label_size), self.start_idx, dtype=torch.int64).to(self.device)
        decodeIdx = torch.LongTensor(batchSize, sentLength).to(self.device)

        scores = all_scores
        scoresRecord[:, 0, :] = scores[:, 0, self.start_idx, :]  ## represent the best current score from the start, is the best

        idxRecord[:,  0, :] = startIds
        for wordIdx in range(1, sentLength):
            ### scoresIdx: batch x from_label x to_label at current index.
            scoresIdx = scoresRecord[:, wordIdx - 1, :].view(batchSize, self.label_size, 1).expand(batchSize, self.label_size,
                                                                                  self.label_size) + scores[:, wordIdx, :, :]
            idxRecord[:, wordIdx, :] = torch.argmax(scoresIdx, 1)  ## the best previous label idx to crrent labels
            scoresRecord[:, wordIdx, :] = torch.gather(scoresIdx, 1, idxRecord[:, wordIdx, :].view(batchSize, 1, self.label_size)).view(batchSize, self.label_size)

        lastScores = torch.gather(scoresRecord, 1, word_seq_lens.view(batchSize, 1, 1).expand(batchSize, 1, self.label_size) - 1).view(batchSize, self.label_size)  ##select position
        lastScores += self.transition[:, self.end_idx].view(1, self.label_size).expand(batchSize, self.label_size)
        decodeIdx[:, 0] = torch.argmax(lastScores, 1)
        bestScores = torch.gather(lastScores, 1, decodeIdx[:, 0].view(batchSize, 1))

        for distance2Last in range(sentLength - 1):
            lastNIdxRecord = torch.gather(idxRecord, 1, torch.where(word_seq_lens - distance2Last - 1 > 0, word_seq_lens - distance2Last - 1, mask).view(batchSize, 1, 1).expand(batchSize, 1, self.label_size)).view(batchSize, self.label_size)
            decodeIdx[:, distance2Last + 1] = torch.gather(lastNIdxRecord, 1, decodeIdx[:, distance2Last].view(batchSize, 1)).view(batchSize)

        return bestScores, decodeIdx

    
    def backward_score(self, emission_scores, seq_lens):
        """backward algorithm"""
        device = emission_scores.device
        all_scores = self.calculate_all_scores(emission_scores)

        batch_size = all_scores.size(0)
        seq_len = all_scores.size(1)

        # beta[T] initialized as 0
        beta = torch.zeros(batch_size, seq_len, self.label_size).to(device)
        beta[:, 0, :] += self.transition[:, self.end_idx].view(1, self.label_size)

        # beta stored in reverse order
        # all score at i: phi(from class at L - i - 1, to class at L - i)
        all_scores = tmu.reverse_sequence(all_scores, seq_lens)
        for word_idx in range(1, seq_len):
            # beta[t + 1]: batch_size, t + 1, to label_size
            # indexing tricky here !! and different than the forward algo
            beta_t_ = beta[:, word_idx - 1, :]\
                .view(batch_size, 1, self.label_size)\
                .expand(batch_size, self.label_size, self.label_size)\

            before_log_sum_exp = beta_t_ + all_scores[:, word_idx - 1, :, :]
            beta[:, word_idx, :] = torch.logsumexp(before_log_sum_exp, 2)

        # reverse beta:
        beta = tmu.reverse_sequence(beta, seq_lens)
        return beta
        
    def marginal(self, emission_scores, seq_lens):
        """Marginal distribution with conventional forward-backward"""
        all_scores=self.calculate_all_scores(emission_scores)
        alpha, log_Z = self.forward_unlabeled(all_scores, seq_lens, True)
        beta = self.backward_score(emission_scores, seq_lens)
        log_marginals = alpha + beta - log_Z.unsqueeze(1).unsqueeze(1)
        return log_marginals 
          
