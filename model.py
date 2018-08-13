import torch
from torch import nn
import torch.nn.functional as F


class MatchLSTM(nn.Module):
    def __init__(self, config, word2vec):
        super(MatchLSTM, self).__init__()
        self.config = config

        use_cuda = config.yes_cuda > 0 and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.word_embed = nn.Embedding(len(word2vec), len(word2vec[0]),
                                       padding_idx=0)
        self.word_embed.weight.data.copy_(torch.from_numpy(word2vec))
        self.word_embed.weight.requires_grad = False

        self.w_e = nn.Parameter(torch.Tensor(config.hidden_size))
        nn.init.uniform_(self.w_e)

        self.linear_s = nn.Linear(in_features=config.hidden_size,
                                  out_features=config.hidden_size, bias=False)
        self.linear_t = nn.Linear(in_features=config.hidden_size,
                                  out_features=config.hidden_size, bias=False)
        self.linear_m = nn.Linear(in_features=config.hidden_size,
                                  out_features=config.hidden_size, bias=False)
        self.fc = nn.Linear(in_features=config.hidden_size,
                            out_features=config.num_classes)
        self.init_linears()

        self.lstm_prem = nn.LSTMCell(config.embedding_dim, config.hidden_size)
        self.lstm_hypo = nn.LSTMCell(config.embedding_dim, config.hidden_size)
        self.lstm_match = nn.LSTMCell(2*config.hidden_size, config.hidden_size)

    def init_linears(self):
        nn.init.xavier_uniform_(self.linear_s.weight)
        nn.init.xavier_uniform_(self.linear_t.weight)
        nn.init.xavier_uniform_(self.linear_m.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.uniform_(self.fc.bias)

    def forward(self, premise, premise_len, hypothesis, hypothesis_len):

        # (max_len, batch_size) -> (max_len, batch_size, embed_dim)
        premise = self.word_embed(premise.to(self.device))
        hypothesis = self.word_embed(hypothesis.to(self.device))

        prem_max_len = premise.size(0)
        batch_size = premise.size(1)

        # premise
        h_s = torch.zeros((prem_max_len, batch_size, self.config.hidden_size),
                          device=self.device)
        for j, premise_j in enumerate(premise):
            h_s_j, _ = self.lstm_prem(premise_j)
            # TODO masking
            h_s[j] = h_s_j

        # h_m_{k-1}
        h_m_km1 = torch.zeros((batch_size, self.config.hidden_size),
                              device=self.device)
        h_m_k = None

        # hypothesis and matchLSTM
        for k, hypothesis_k in enumerate(hypothesis):
            h_t_k, _ = self.lstm_hypo(hypothesis_k)
            # TODO masking

            # Equation (6)
            e_kj = torch.zeros((prem_max_len, batch_size), device=self.device)
            for j in range(prem_max_len):
                for l in range(batch_size):
                    e_kj[j][l] = \
                        torch.dot(self.w_e,
                                  torch.tanh(self.linear_s(h_s[j][l]) +
                                             self.linear_t(h_t_k[l]) +
                                             self.linear_m(h_m_km1[l])))

            # Equation (3)
            # (max_len, batch_size)
            alpha_kj = F.softmax(e_kj, dim=0)

            # Equation (2)
            # (batch_size, hidden_size)
            a_k = torch.zeros((batch_size, self.config.hidden_size),
                              device=self.device)
            for l in range(batch_size):
                for j in range(prem_max_len):
                    # alpha_h
                    a_k[l] += alpha_kj[j][l] * h_s[j][l]

            # Equation (7)
            # (batch_size, 2 * hidden_size)
            m_k = torch.cat((a_k, h_t_k), 1)

            # Equation (8)
            h_m_k, _ = self.lstm_match(m_k)

            h_m_km1 = h_m_k

        return self.fc(h_m_k)

    def get_req_grad_params(self, debug=False):
        print('#parameters: ', end='')
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
        print('{:,}'.format(total_size))
        return params
