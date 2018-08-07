import corenlp
import os
import spacy
import torch
from torch.utils.data import Dataset


class SNLIData(object):
    def __init__(self, config):
        self.config = config
        self.num_classes = config.num_classes

        self.ngram2idx = dict()
        self.idx2ngram = dict()
        self.ngram2idx['PAD'] = 0
        self.idx2ngram[0] = 'PAD'
        self.ngram2idx['UNK'] = 1
        self.idx2ngram[1] = 'UNK'

        # label num order
        self.label_dict = {'entailment': 0, 'contradiction': 1, 'neutral': 2}

        os.environ['CORENLP_HOME'] = \
            os.path.expanduser('~') + '/common/stanford-corenlp-full-2017-06-09'

        # https://spacy.io/usage/facts-figures#benchmarks-models-english
        # Run the following command on terminal
        # python3 -m spacy download en_core_web_lg
        self.nlp = spacy.load('en_core_web_lg',
                              disable=['parser', 'tagger', 'ner'])
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))

        self.word_dict = self.get_word_dict()
        print('word_dict size', len(self.word_dict))

        print('Loading GloVe .. {}'.format(self.config.glove_path))
        self.word2vec = dict()
        with open(self.config.glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                cols = line.split(' ')
                if cols[0] in self.word_dict:
                    self.word2vec[cols[0]] = [float(l) for l in cols[1:]]

        print('word2vec len', len(self.word2vec))

        self.train_data, self.valid_data, self.test_data = self.get_data()

    def get_word_dict(self):
        word_dict = dict()

        with corenlp.CoreNLPClient(annotators="tokenize ssplit".split()) as \
                client:
            with open(self.config.train_data_path, 'r', newline='',
                      encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    # skip the first line
                    if idx == 0:
                        continue

                    cols = line.split('\t')
                    # print(cols)

                    if cols[0] == '-':
                        continue

                    # premise = cols[5]
                    # premise_sentence = client.annotate(premise).sentence[0]
                    # # assert corenlp.to_text(premise_sentence) == premise
                    # if corenlp.to_text(premise_sentence) != premise:
                    #     print(corenlp.to_text(premise_sentence), premise)
                    # premise_words = [t.originalText
                    #                  for t in premise_sentence.token]
                    #
                    # hypothesis = cols[6]
                    # hypothesis_sentence = \
                    #     client.annotate(hypothesis).sentence[0]
                    # # assert corenlp.to_text(hypothesis_sentence) == hypothesis
                    # if corenlp.to_text(hypothesis_sentence) != hypothesis:
                    #     print(corenlp.to_text(hypothesis_sentence), hypothesis)
                    # hypothesis_words = [t.originalText
                    #                     for t in hypothesis_sentence.token]

                    # for w in (premise_words + hypothesis_words):
                    for w in (cols[5].split(' ') + cols[6].split(' ')):
                        if w in word_dict:
                            word_dict[w] += 1
                        else:
                            word_dict[w] = 1

                    if idx + 1 % 10000 == 0:
                        print(idx + 1)

        return word_dict

    def get_data(self):
        train_data = self.load(self.config.train_data_path, is_train=True)
        valid_data = self.load(self.config.valid_data_path)
        test_data = self.load(self.config.test_data_path)
        return train_data, valid_data, test_data

    def load(self, data_path, is_train=False):
        data = list()
        with open(data_path, 'r', newline='', encoding='utf-8') as f:
            for idx, line in enumerate(f):

                # skip the first line
                if idx == 0:
                    continue

                cols = line.split('\t')
                # print(cols)

                if cols[0] == '-':
                    continue

                y = self.label_dict[cols[0]]

                premise_doc = self.nlp(cols[5])
                premise_words = [token.text for token in premise_doc]

                hypothesis_doc = self.nlp(cols[6])
                hypothesis_words = [token.text for token in hypothesis_doc]

                if is_train:
                    for w in (premise_words + hypothesis_words):
                        idx = self.ngram2idx.get(w)
                        if idx is None:
                            idx = len(self.ngram2idx)
                            self.ngram2idx[w] = idx
                            self.idx2ngram[idx] = w

                premise = [self.ngram2idx[w] if w in self.ngram2idx
                           else self.ngram2idx['UNK']
                           for w in premise_words]

                hypothesis = [self.ngram2idx[w] if w in self.ngram2idx
                              else self.ngram2idx['UNK']
                              for w in hypothesis_words]

                data.append([premise, hypothesis, y])

                if (idx + 1) % 10000 == 0:
                    print(idx + 1)

        if is_train:
            print('dictionary size', len(self.ngram2idx))

        return data

    def get_dataloaders(self, batch_size=32, shuffle=True, num_workers=4):
        train_loader = torch.utils.data.DataLoader(
            SNLIDataset(self.train_data),
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=self.batchify,
            pin_memory=True
        )

        valid_loader = torch.utils.data.DataLoader(
            SNLIDataset(self.valid_data),
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=self.batchify,
            pin_memory=True
        )

        test_loader = torch.utils.data.DataLoader(
            SNLIDataset(self.test_data),
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=self.batchify,
            pin_memory=True
        )
        return train_loader, valid_loader, test_loader

    @staticmethod
    def batchify(b):
        x = [e[0] for e in b]
        y = [e[2] for e in b]

        x = torch.tensor(x, dtype=torch.int64)
        y = torch.tensor(y, dtype=torch.int64)

        return x, y


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
    parser.add_argument('--seed', type=int, default=2018)
    parser.add_argument('--num_classes', type=int, default=3)
    args = parser.parse_args()

    pprint.PrettyPrinter().pprint(args.__dict__)

    import os
    if os.path.exists(args.pickle_path):
        with open(args.pickle_path, 'rb') as f_pkl:
            snlidata = pickle.load(f_pkl)
    else:
        snlidata = SNLIData(args)
    #     with open(args.pickle_path, 'wb') as f_pkl:
    #         pickle.dump(snlidata, f_pkl)
    #
    # tr_loader, _, _ = snlidata.get_dataloaders(batch_size=256, num_workers=4)
    # # print(len(tr_loader.dataset))
    # for batch_idx, batch in enumerate(tr_loader):
    #     if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(tr_loader):
    #         print(datetime.now(), 'batch', batch_idx + 1)
