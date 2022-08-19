
import torch
import torch.nn as nn

# from datastruct import ContextEmb
from encoder.charbilstm import CharBiLSTM
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from overrides import overrides

class BiLSTMEncoder(nn.Module):

    def __init__(self, config, print_info: bool = True):
        """
        param
            config: a class that contains all the configurations
            print_info: whether print some information or not
        """
        super(BiLSTMEncoder, self).__init__()

        self.label_size = config.label_size
        self.device = config.device
        self.use_char = config.use_char_rnn
        # self.context_emb = config.context_emb

        self.label2idx = config.label2idx
        self.labels = config.idx2label

        self.input_size = config.embedding_dim

        self.ent_freq = config.ent_freq
        self.tau = config.tau
        
        # initialize the embedding from pretrained model, if None, then use random initialization
        self.word_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(config.word_embedding), freeze=False).to(self.device)
        self.word_drop = nn.Dropout(config.dropout).to(self.device)

        if print_info:
            print("[Model Info] Input size to LSTM: {}".format(self.input_size))
            print("[Model Info] LSTM Hidden Size: {}".format(config.hidden_dim))

        self.lstm = nn.LSTM(self.input_size, config.hidden_dim // 2, num_layers=1, batch_first=True, bidirectional=True).to(self.device)

        self.drop_lstm = nn.Dropout(config.dropout).to(self.device)

        final_hidden_dim = config.hidden_dim

        if print_info:
            print("[Model Info] Final Hidden Size: {}".format(final_hidden_dim))
        
        self.hidden2tag = nn.Linear(final_hidden_dim, self.label_size).to(self.device)
        self.sigmoid = nn.Sigmoid().to(self.device)

    @overrides
    def forward(self, input_ids: torch.Tensor,
                      word_seq_lens: torch.Tensor,
                      attention_mask: torch.Tensor,
                      token2word_ids: torch.Tensor) -> torch.Tensor:
        """
        Encoding the input with BiLSTM
        param 
           input_ids: (batch_size, sent_len) 
           word_seq_lens: (batch_size, 1)
           attention_mask: not used here, keep just for unified forward call
           token2word_ids: not used here, keep just for unified forward call
        return 
           emission scores (batch_size, sent_len, label_size)
        """

        word_emb = self.word_embedding(input_ids)
        
        # if self.context_emb != ContextEmb.none:
        #     word_emb = torch.cat([word_emb, batch_context_emb.to(self.device)], 2)
        
        # if self.use_char:
        #     char_features = self.char_feature(char_inputs, char_seq_lens)
        #     word_emb = torch.cat([word_emb, char_features], 2)

        word_rep = self.word_drop(word_emb)

        sorted_seq_len, permIdx = word_seq_lens.sort(0, descending=True)
        _, recover_idx = permIdx.sort(0, descending=False)
        sorted_seq_tensor = word_rep[permIdx]

        packed_words = pack_padded_sequence(sorted_seq_tensor, sorted_seq_len.cpu(), True)
        lstm_out, _ = self.lstm(packed_words, None)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        feature_out = self.drop_lstm(lstm_out)
        outputs = self.hidden2tag(feature_out)
        
        outputs = outputs[recover_idx]
        if self.tau>0:
            outputs = outputs + self.tau * ((torch.tensor(self.ent_freq, dtype=torch.long) + 1e-12).log()).to(self.device)

        return outputs, 0





        