import torch
from torch import nn
import torch.nn.functional as F


class MatchLSTM(nn.Module):
    def __init__(self, config, word2vec):
        super(MatchLSTM, self).__init__()
        self.config = config

        self.word_embed = nn.Embedding(len(word2vec), len(word2vec[0]),
                                       padding_idx=0)
        self.word_embed.weight.data.copy_(torch.from_numpy(word2vec))
        self.word_embed.weight.requires_grad = False

        self.w_e = torch.zeros(config.hidden_size)
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

        self.lstm_prem = nn.LSTMCell(config.input_size, config.hidden_size)
        self.lstm_hypo = nn.LSTMCell(config.input_size, config.hidden_size)
        self.lstm_match = nn.LSTMCell(2*config.hidden_size, config.hidden_size)

    def init_linears(self):
        nn.init.xavier_uniform_(self.linear_s.weight)
        nn.init.xavier_uniform_(self.linear_t.weight)
        nn.init.xavier_uniform_(self.linear_m.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.uniform_(self.fc.bias)

    def forward(self, premise_tpl, hypothesis_tpl):
        premise, premise_len = premise_tpl
        hypothesis, hypothesis_len = hypothesis_tpl

        # (batch_size, max_len) -> (batch_size, max_len, embed_dim)
        premise_embed = self.word_embed(premise)
        hypothesis_embed = self.word_embed(hypothesis)

        batch_size = premise_embed.size(0)

        outputs = torch.zeros((batch_size, self.config.num_classes))

        for i, (prem_emb, prem_len, hypo_emb, hypo_len) in \
                enumerate(zip(premise_embed, premise_len,
                              hypothesis_embed, hypothesis_len)):

            h_s, _ = self.lstm_prem(prem_emb[:prem_len.item()])

            # h_m_{k-1}
            h_m_km1 = torch.zeros(self.config.hidden_size)
            h_m_k = None

            h_t, _ = self.lstm_hypo(hypo_emb[:hypo_len.item()])

            for k in range(hypo_len.item()):
                h_t_k = h_t[k]

                # Equation (6)
                e_kj_tensor = torch.zeros(prem_len.item())
                for j in range(prem_len.item()):
                    e_kj = torch.dot(self.w_e,
                                     torch.tanh(self.linear_s(h_s[j]) +
                                                self.linear_t(h_t_k) +
                                                self.linear_m(h_m_km1)))
                    e_kj_tensor[j] = e_kj

                # Equation (3)
                alpha_kj = F.softmax(e_kj_tensor, dim=0)

                # Equation (2)
                a_k = torch.zeros(self.config.hidden_size)
                for j in range(prem_len.item()):
                    alpha_h = alpha_kj[j] * h_s[j]
                    for idx in range(self.config.hidden_size):
                        a_k[idx] += alpha_h[idx]  # element-wise sum

                # Equation (7)
                m_k = torch.cat((a_k, h_t_k), 0)

                # Equation (8)
                h_m_k, _ = self.lstm_match(torch.unsqueeze(m_k, 0))

                h_m_km1 = h_m_k[0]

            outputs[i] = self.fc(h_m_k[0])

        return F.softmax(outputs, dim=1)

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
