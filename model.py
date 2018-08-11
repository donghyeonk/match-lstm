# import numpy as np
import torch
from torch import nn
# import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class MatchLSTM(nn.Module):
    def __init__(self, config, word2vec):
        super(MatchLSTM, self).__init__()
        self.config = config

        self.word_embed = nn.Embedding(len(word2vec), len(word2vec[0]))
        self.word_embed.weight.data.copy_(torch.from_numpy(word2vec))
        self.word_embed.weight.requires_grad = False

        self.linear_s = nn.Linear(in_features=config.hidden_size,
                                  out_features=config.hidden_size, bias=False)
        self.linear_t = nn.Linear(in_features=config.hidden_size,
                                  out_features=config.hidden_size, bias=False)
        self.linear_m = nn.Linear(in_features=config.hidden_size,
                                  out_features=config.hidden_size, bias=False)

        self.batch_first = True

        # premise
        self.lstm_s = nn.LSTM(config.input_size, config.hidden_size,
                              config.num_layers, batch_first=self.batch_first)
        # hypothesis
        self.lstm_t = nn.LSTM(config.input_size, config.hidden_size,
                              config.num_layers, batch_first=self.batch_first)

        # attention
        self.lstm_m = nn.LSTM(2 * config.hidden_size, config.hidden_size,
                              config.num_layers, batch_first=self.batch_first)

    def forward(self, premise_tpl, hypothesis_tpl):
        premise, premise_len = premise_tpl
        premise_embed = self.word_embed(premise)
        premise_len, premise_perm_idxes = \
            premise_len.sort(dim=0, descending=True)
        premise_embed = premise_embed[premise_perm_idxes]
        _, premise_idx_unsort = \
            torch.sort(premise_perm_idxes, dim=0, descending=False)
        premise_input = \
            pack_padded_sequence(premise_embed, premise_len,
                                 batch_first=self.batch_first)
        premise_output, (h_s, c_s) = self.lstm_s(premise_input)

        premise_output, _ = pad_packed_sequence(premise_output,
                                                batch_first=self.batch_first)
        premise_output = premise_output[premise_idx_unsort]
        premise_len = premise_len[premise_idx_unsort]

        # self.linear_s(h_s)

        # TODO h_t
        # TODO h_m
        # TODO e, a

    def get_req_grad_params(self, debug=False):
        print('model parameters: ', end='')
        params = list()
        total_size = 0

        def multiply_iter(p_list):
            out = 1
            for _p in p_list:
                out *= _p
            return out

        for p in self.parameters():
            if p.requires_grad:
                params.append(p)
                total_size += multiply_iter(p.size())
            if debug:
                print(p.requires_grad, p.size())
        print('%s' % '{:,}'.format(total_size))
        return params
