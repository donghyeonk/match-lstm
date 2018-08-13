import numpy as np
import torch
from torch.utils.data import Dataset


class SNLIData(object):
    def __init__(self, config):
        self.config = config

        # label num order
        self.label_dict = {'entailment': 0, 'contradiction': 1, 'neutral': 2}

        self.word2idx = dict()
        self.idx2word = dict()

        self.pad = '<PAD>'
        self.word2idx[self.pad] = 0
        self.idx2word[0] = self.pad

        self.null_word = '<NULL>'
        self.word2idx[self.null_word] = 1
        self.idx2word[1] = self.null_word

        self.build_word_set()
        print('#SNLI words', len(self.word2idx))

        self.word_embeds = self.get_glove()
        self.word_embeds[self.pad] = [0.] * self.config.embedding_dim
        self.word_embeds[self.null_word] = [0.] * self.config.embedding_dim
        print('SNLI - GloVe intersection size', len(self.word_embeds),
              '({:.1f}%)'.format(100*len(self.word_embeds)/len(self.word2idx)))

        self.unseen_word_dict = dict()
        self.unseen_word_count_dict = dict()

        self.premise_max_len = 0
        self.hypothesis_max_len = 0

        self.train_data, self.valid_data, self.test_data = self.get_split_data()
        assert len(self.train_data) == 549367
        assert len(self.valid_data) == 9842
        assert len(self.test_data) == 9824

        print('train', len(self.train_data))
        print('dev  ', len(self.valid_data))
        print('test ', len(self.test_data))

        print('#word_embeddings', len(self.word_embeds))
        print('premise_max_len', self.premise_max_len)
        print('hypothesis_max_len', self.hypothesis_max_len)

        self.word2vec = np.zeros((len(self.word_embeds),
                                  self.config.embedding_dim))
        for idx in self.idx2word:
            self.word2vec[idx] = self.word_embeds[self.idx2word[idx]]

        assert len(self.word_embeds) == len(self.word2idx) == len(self.word2vec)

        assert self.premise_max_len == config.premise_max_len
        assert self.hypothesis_max_len == config.hypothesis_max_len

    def build_word_set(self):
        def update_dict(data_path):
            with open(data_path, 'r', newline='', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    # skip the first line
                    if idx == 0:
                        continue
                    cols = line.rstrip().split('\t')
                    # print(cols)

                    if cols[0] == '-':
                        continue

                    premise = [w for w in cols[1].split(' ')
                               if w != '(' and w != ')']
                    hypothesis = [w for w in cols[2].split(' ')
                                  if w != '(' and w != ')']
                    for w in premise+hypothesis:
                        if w not in self.word2idx:
                            idx = len(self.word2idx)
                            self.word2idx[w] = idx
                            self.idx2word[idx] = w

                    # for uw in ['https://www.youtube.com/watch?v=tXrsvC25GH8',
                    #            'cantunderstans',
                    #            'http://fresnosmilemakeovers.com/',
                    #            'motocyckes',
                    #            'arefun']:
                    #     if uw in premise or uw in hypothesis:
                    #         print(data_path, cols[8], premise, hypothesis)

        update_dict(self.config.train_data_path)
        update_dict(self.config.valid_data_path)
        update_dict(self.config.test_data_path)

    def get_glove(self):
        print('Loading GloVe .. {}'.format(self.config.glove_path))
        word2vec = dict()
        with open(self.config.glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                cols = line.split(' ')
                if cols[0] in self.word2idx:
                    word2vec[cols[0]] = [float(l) for l in cols[1:]]
        return word2vec

    def get_split_data(self):
        train_data = self.load(self.config.train_data_path)
        valid_data = self.load(self.config.valid_data_path)
        test_data = self.load(self.config.test_data_path)

        print('#unseen_words', len(self.unseen_word_dict))

        # an approximation of the embedding of an unseen word
        for w in self.unseen_word_dict:
            if w in self.unseen_word_count_dict:
                self.unseen_word_dict[w] /= self.unseen_word_count_dict[w]
            else:
                print('all-zero unseen word', w)
            self.word_embeds[w] = self.unseen_word_dict[w]

        return train_data, valid_data, test_data

    def load(self, data_path):
        data = list()

        null_word_idx = self.word2idx[self.null_word]
        pad_idx = self.word2idx[self.pad]

        def approximate_unseen(sentence):
            sentence_len = len(sentence)
            for w_idx, w in enumerate(sentence):
                if w not in self.word_embeds:
                    if w not in self.unseen_word_dict:
                        self.unseen_word_dict[w] = \
                            np.zeros(self.config.embedding_dim)
                    for r in range(-self.config.window_size,
                                   self.config.window_size + 1):
                        if r != 0 and 0 <= w_idx + r < sentence_len \
                                and sentence[w_idx + r] in self.word_embeds:
                            self.unseen_word_dict[w] = \
                                np.add(self.unseen_word_dict[w],
                                       self.word_embeds[sentence[w_idx + r]])
                            if w in self.unseen_word_count_dict:
                                self.unseen_word_count_dict[w] += 1
                            else:
                                self.unseen_word_count_dict[w] = 1

        with open(data_path, 'r', newline='', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                # skip the first line
                if idx == 0:
                    continue

                cols = line.rstrip().split('\t')
                # print(cols)

                if cols[0] == '-':
                    continue

                premise = [w for w in cols[1].split(' ')
                           if w != '(' and w != ')']
                hypothesis = [w for w in cols[2].split(' ')
                              if w != '(' and w != ')']
                y = self.label_dict[cols[0]]

                approximate_unseen(premise)
                approximate_unseen(hypothesis)

                premise_idx = [self.word2idx[w] for w in premise]
                premise_idx.append(null_word_idx)  # NULL
                hypothesis_idx = [self.word2idx[w] for w in hypothesis]

                premise_len = len(premise_idx)
                hypothesis_len = len(hypothesis_idx)

                # padding
                while len(premise_idx) < self.config.premise_max_len:
                    premise_idx.append(pad_idx)
                assert len(premise_idx) == self.config.premise_max_len
                while len(hypothesis_idx) < self.config.hypothesis_max_len:
                    hypothesis_idx.append(pad_idx)
                assert len(hypothesis_idx) == self.config.hypothesis_max_len

                data.append([premise_idx, premise_len,
                             hypothesis_idx, hypothesis_len,
                             y])

                if self.premise_max_len < premise_len:
                    self.premise_max_len = premise_len

                if self.hypothesis_max_len < hypothesis_len:
                    self.hypothesis_max_len = hypothesis_len

                # if (idx + 1) % 100000 == 0:
                #     print(idx + 1)

        return data

    def get_dataloaders(self, batch_size=32, shuffle=True, num_workers=4,
                        pin_memory=True):
        train_loader = torch.utils.data.DataLoader(
            SNLIDataset(self.train_data),
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
            # collate_fn=self.batchify,
            pin_memory=pin_memory
        )

        valid_loader = torch.utils.data.DataLoader(
            SNLIDataset(self.valid_data),
            batch_size=batch_size,
            num_workers=num_workers,
            # collate_fn=self.batchify,
            pin_memory=pin_memory
        )

        test_loader = torch.utils.data.DataLoader(
            SNLIDataset(self.test_data),
            batch_size=batch_size,
            num_workers=num_workers,
            # collate_fn=self.batchify,
            pin_memory=pin_memory
        )
        return train_loader, valid_loader, test_loader

    def batchify(self, b):
        pad_idx = self.word2idx[self.pad]

        premise = [e[0] for e in b]
        premise_len = [e[1] for e in b]
        premise_len_max = max(premise_len)
        premise = torch.tensor([p + [pad_idx] * (premise_len_max - len(p))
                                for p in premise], dtype=torch.int64)
        premise_len = torch.tensor(premise_len, dtype=torch.int64)

        hypothesis = [e[2] for e in b]
        hypothesis_len = [e[3] for e in b]
        hypothesis_len_max = max(hypothesis_len)
        hypothesis = torch.tensor([h + [pad_idx] * (hypothesis_len_max - len(h))
                                   for h in hypothesis], dtype=torch.int64)
        hypothesis_len = torch.tensor(hypothesis_len, dtype=torch.int64)

        y = torch.tensor([e[4] for e in b], dtype=torch.int64)

        return (premise, premise_len), (hypothesis, hypothesis_len), y


class SNLIDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


if __name__ == '__main__':
    import argparse
    from datetime import datetime
    import os
    import pickle
    import pprint

    home_dir = os.path.expanduser('~')
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str,
                        default='./data/snli_1.0_train.txt')
    parser.add_argument('--valid_data_path', type=str,
                        default='./data/snli_1.0_dev.txt')
    parser.add_argument('--test_data_path', type=str,
                        default='./data/snli_1.0_test.txt')
    parser.add_argument('--glove_path', type=str,
                        default=home_dir + '/common/glove/glove.840B.300d.txt')
    parser.add_argument('--pickle_path', type=str, default='./data/snli.pkl')
    parser.add_argument('--embedding_dim', type=int, default=300)
    parser.add_argument('--window_size', type=int, default=4)
    parser.add_argument('--premise_max_len', type=int, default=83)
    parser.add_argument('--hypothesis_max_len', type=int, default=62)
    parser.add_argument('--overwrite', type=int, default=1)
    args = parser.parse_args()

    pprint.PrettyPrinter().pprint(args.__dict__)

    if os.path.exists(args.pickle_path) and args.overwrite == 0:
        print('Found', args.pickle_path)
        with open(args.pickle_path, 'rb') as f_pkl:
            snlidata = pickle.load(f_pkl)
    else:
        snlidata = SNLIData(args)
        with open(args.pickle_path, 'wb') as f_pkl:
            pickle.dump(snlidata, f_pkl)

    tr_loader, _, _ = \
        snlidata.get_dataloaders(batch_size=256, num_workers=4,
                                 pin_memory=torch.cuda.is_available())
    # print(len(tr_loader.dataset))
    for batch_idx, batch in enumerate(tr_loader):
        if (batch_idx + 1) % 1000 == 0 or (batch_idx + 1) == len(tr_loader):
            print(datetime.now(), 'batch', batch_idx + 1)
