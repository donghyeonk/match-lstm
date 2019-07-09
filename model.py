import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class MatchLSTM(nn.Module):
    def __init__(self, config, word2vec):
        super(MatchLSTM, self).__init__()
        self.config = config

        use_cuda = config.yes_cuda > 0 and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        print('word2vec shape', word2vec.shape)
        assert len(word2vec[0]) == config.embedding_dim
        self.word_embed = nn.Embedding(len(word2vec), len(word2vec[0]),
                                       padding_idx=0)
        self.word_embed.weight.data.copy_(torch.from_numpy(word2vec))
        self.word_embed.weight.requires_grad = False

        # TODO emb_partial update
        # https://github.com/shuohangwang/SeqMatchSeq/blob/master/main/main.lua#L42

        self.w_e = nn.Parameter(torch.Tensor(config.hidden_size))
        nn.init.uniform_(self.w_e)

        self.w_s = nn.Linear(in_features=config.hidden_size,
                             out_features=config.hidden_size, bias=False)
        self.w_t = nn.Linear(in_features=config.hidden_size,
                             out_features=config.hidden_size, bias=False)
        self.w_m = nn.Linear(in_features=config.hidden_size,
                             out_features=config.hidden_size, bias=False)
        self.fc = nn.Linear(in_features=config.hidden_size,
                            out_features=config.num_classes)
        self.init_linears()

        self.lstm_prem = nn.LSTM(config.embedding_dim, config.hidden_size)
        self.lstm_hypo = nn.LSTM(config.embedding_dim, config.hidden_size)
        self.lstm_match = nn.LSTMCell(2 * config.hidden_size,
                                      config.hidden_size)

        if config.dropout_fc > 0.:
            self.dropout_fc = nn.Dropout(p=config.dropout_fc)

        self.req_grad_params = self.get_req_grad_params(debug=True)

    def init_linears(self):
        nn.init.xavier_uniform_(self.w_s.weight)
        nn.init.xavier_uniform_(self.w_t.weight)
        nn.init.xavier_uniform_(self.w_m.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, premise, premise_len, hypothesis, hypothesis_len):
        # premise
        premise = premise.to(self.device)
        prem_max_len = premise.size(0)
        premise_len, p_idxes = premise_len.sort(dim=0, descending=True)
        _, p_idx_unsort = torch.sort(p_idxes, dim=0, descending=False)
        premise = premise[:, p_idxes]
        # (max_len, batch_size) -> (max_len, batch_size, embed_dim)
        if self.config.dropout_emb > 0. and self.training:
            premise = F.dropout(self.word_embed(premise),
                                p=self.config.dropout_emb,
                                training=self.training)
        else:
            premise = self.word_embed(premise)
        packed_premise = pack_padded_sequence(premise, premise_len)
        # (max_len, batch_size, hidden_size)
        h_s, (_, _) = self.lstm_prem(packed_premise)
        h_s, _ = pad_packed_sequence(h_s)
        h_s = h_s[:, p_idx_unsort]

        # hypothesis
        hypothesis = hypothesis.to(self.device)
        hypothesis_max_len = hypothesis.size(0)
        hypothesis_len, h_idxes = hypothesis_len.sort(dim=0, descending=True)
        _, h_idx_unsort = torch.sort(h_idxes, dim=0, descending=False)
        hypothesis = hypothesis[:, h_idxes]
        # (max_len, batch_size) -> (max_len, batch_size, embed_dim)
        if self.config.dropout_emb > 0. and self.training:
            hypothesis = F.dropout(self.word_embed(hypothesis),
                                   p=self.config.dropout_emb,
                                   training=self.training)
        else:
            hypothesis = self.word_embed(hypothesis)
        packed_hypothesis = pack_padded_sequence(hypothesis, hypothesis_len)
        # (max_len, batch_size, hidden_size)
        h_t, (_, _) = self.lstm_hypo(packed_hypothesis)
        h_t, _ = pad_packed_sequence(h_t)
        h_t = h_t[:, h_idx_unsort]
        hypothesis_len = hypothesis_len[h_idx_unsort]

        # matchLSTM
        batch_size = premise.size(1)
        h_m_k = torch.zeros((batch_size, self.config.hidden_size),
                            device=self.device)
        c_m_k = torch.zeros((batch_size, self.config.hidden_size),
                            device=self.device)
        h_last = torch.zeros((batch_size, self.config.hidden_size),
                             device=self.device)

        for k in range(hypothesis_max_len):
            h_t_k = h_t[k]

            # Equation (6)
            # e_kj: (prem_max_len, batch_size)
            e_kj = torch.zeros((prem_max_len, batch_size), device=self.device)
            w_e_expand = self.w_e.expand(batch_size, self.config.hidden_size)
            for j in range(prem_max_len):
                # tanh_stm: (batch_size, hidden_size)
                tanh_s_t_m = torch.tanh(
                    self.w_s(h_s[j]) + self.w_t(h_t_k) + self.w_m(h_m_k))

                # dot product
                # https://github.com/pytorch/pytorch/issues/18027#issuecomment-473404765
                e_kj[j] = (w_e_expand * tanh_s_t_m).sum(-1)

            # Equation (3)
            # (prem_max_len, batch_size)
            alpha_kj = F.softmax(e_kj, dim=0)

            # Equation (2)
            # (batch_size, hidden_size)
            a_k = torch.bmm(
                torch.unsqueeze(alpha_kj.t(), 1), h_s.permute(1, 0, 2))
            a_k = torch.squeeze(a_k, dim=1)

            # Equation (7)
            # (batch_size, 2 * hidden_size)
            m_k = torch.cat((a_k, h_t_k), 1)

            # Equation (8)
            # (batch_size, hidden_size)
            h_m_k, c_m_k = self.lstm_match(m_k, (h_m_k, c_m_k))

            # handle variable length sequences: hypothesis
            # (batch_size)
            for batch_idx, hl in enumerate(hypothesis_len):
                if k + 1 == hl:
                    h_last[batch_idx] = h_m_k[batch_idx]

        if self.config.dropout_fc > 0:
            h_last = self.dropout_fc(h_last)

        return self.fc(h_last)

    def get_req_grad_params(self, debug=False):
        print('#parameters: ', end='')
        params = list()
        total_size = 0

        def multiply_iter(p_list):
            out = 1
            for _p in p_list:
                out *= _p
            return out

        for name, p in self.named_parameters():
            if p.requires_grad:
                params.append(p)
                n_params = multiply_iter(p.size())
                total_size += n_params
            if debug:
                print(name, p.requires_grad, p.size(), multiply_iter(p.size()),
                      sep='\t')
        print('{:,}'.format(total_size))
        return params
