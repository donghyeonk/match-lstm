import argparse
from datetime import datetime
import os
import pickle
import pprint
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from dataset import SNLIData
from model import MatchLSTM

# Ref.
# https://github.com/shuohangwang/SeqMatchSeq/blob/master/main/main.lua
parser = argparse.ArgumentParser()
parser.add_argument('--name', default="mLSTM")
parser.add_argument('--seed', type=int, default=2019)
parser.add_argument('--data_path', type=str, default='./data/snli.pkl')
parser.add_argument('--num_classes', type=int, default=3)

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_decay', type=float, default=0.95)
parser.add_argument('--grad_max_norm', type=float, default=0.)  #

parser.add_argument('--embedding_dim', type=int, default=300)
parser.add_argument('--hidden_size', type=int, default=300)

parser.add_argument('--dropout_fc', type=float, default=0.)  #
parser.add_argument('--dropout_emb', type=float, default=0.3)

parser.add_argument('--batch_size', type=int, default=30)
parser.add_argument('--epochs', type=int, default=10)

parser.add_argument('--log_interval', type=int, default=500)
parser.add_argument('--yes_cuda', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--num_processes', type=int, default=2)
parser.add_argument('--test', type=int, default=0)


def train_mp(rank, device, snli_dataset, model, loss_func, config):
    torch.manual_seed(config.seed + rank)

    test_loader = None
    if config.test == 0:
        train_loader, dev_loader = snli_dataset.get_train_dev_loader(
            batch_size=config.batch_size, num_workers=config.num_workers,
            pin_memory='cuda' == device.type)
    else:
        train_loader, dev_loader, test_loader = snli_dataset.get_dataloaders(
            batch_size=config.batch_size, num_workers=config.num_workers,
            pin_memory='cuda' == device.type)

    optimizer = optim.Adam(model.req_grad_params, lr=config.lr,
                           betas=(0.9, 0.999), amsgrad=True)

    best_loss = float('inf')
    best_acc = 0.
    best_epoch = 0

    for epoch in range(1, config.epochs + 1):
        train_epoch(device, train_loader, model, epoch, optimizer, loss_func,
                    config)

        dev_loss, dev_acc = \
            evaluate_epoch(device, dev_loader, model, epoch, loss_func, 'Dev')

        if dev_loss < best_loss:
            best_loss = dev_loss
            best_acc = dev_acc
            best_epoch = epoch
            save_model(model, optimizer, config,
                       os.path.join('./ckpt', '{}.pth'.format(config.name)))

        print('\tLowest Dev Loss {:.6f}, Acc. {:.1f}%, Epoch {}'.
              format(best_loss, 100 * best_acc, best_epoch))

        # Learning rate decay
        for param_group in optimizer.param_groups:
            print('lr: {:.6f} -> {:.6f}'
                  .format(param_group['lr'],
                          param_group['lr'] * config.lr_decay))
            param_group['lr'] *= config.lr_decay

        if config.test != 0:
            evaluate_epoch(device, test_loader, model, epoch, loss_func, 'Test')


def train_epoch(device, loader, model, epoch, optimizer, loss_func, config):
    model.train()
    train_loss = 0.
    example_count = 0
    correct = 0
    start_t = datetime.now()
    pid = os.getpid()
    for batch_idx, ex in enumerate(loader):
        target = ex[4].to(device)
        optimizer.zero_grad()
        output = model(ex[0], ex[1], ex[2], ex[3])
        loss = loss_func(output, target)
        loss.backward()
        if config.grad_max_norm > 0.:
            torch.nn.utils.clip_grad_norm_(model.req_grad_params,
                                           config.grad_max_norm)
        optimizer.step()

        batch_loss = len(output) * loss.item()
        train_loss += batch_loss
        example_count += len(target)

        pred = torch.max(output, 1)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()

        if (batch_idx + 1) % config.log_interval == 0 \
                or batch_idx == len(loader) - 1:
            _progress = \
                '{}\tpid {}, Train Epoch {}, [{}/{} ({:.1f}%)],' \
                ' Batch Loss: {:.6f}' \
                .format(datetime.now(), pid, epoch,
                        example_count, len(loader.dataset),
                        100. * example_count / len(loader.dataset),
                        batch_loss / len(output))
            print(_progress)

    train_loss /= len(loader.dataset)
    acc = correct / len(loader.dataset)
    print('{} Train Epoch {}, Avg. Loss: {:.6f}, Accuracy: {}/{} ({:.1f}%)'.
          format(datetime.now()-start_t, epoch, train_loss,
                 correct, len(loader.dataset), 100. * acc))
    return train_loss


def evaluate_epoch(device, loader, model, epoch, loss_func, mode):
    model.eval()
    eval_loss = 0.
    correct = 0
    start_t = datetime.now()
    with torch.no_grad():
        for batch_idx, ex in enumerate(loader):
            target = ex[4].to(device)
            output = model(ex[0], ex[1], ex[2], ex[3])
            loss = loss_func(output, target)
            eval_loss += len(output) * loss.item()
            pred = torch.max(output, 1)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    eval_loss /= len(loader.dataset)
    acc = correct / len(loader.dataset)
    print('{} {} Epoch {}, Avg. Loss: {:.6f}, '
          'Accuracy: {}/{} ({:.1f}%)'.format(datetime.now() - start_t, mode,
                                             epoch, eval_loss,
                                             correct, len(loader.dataset),
                                             100. * acc))
    return eval_loss, acc


def save_model(model, optimizer, args, model_save_path):
    # save a model and args
    model_dict = dict()
    model_dict['state_dict'] = model.state_dict()
    model_dict['m_config'] = args
    model_dict['optimizer'] = optimizer.state_dict()
    if not os.path.exists(os.path.dirname(model_save_path)):
        os.makedirs(os.path.dirname(model_save_path))
    torch.save(model_dict, model_save_path)
    print('Saved', model_save_path)


def load_model(model, load_path):
    print('Load checkpoint', load_path)
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['state_dict'])


def main():
    args = parser.parse_args()
    pprint.PrettyPrinter().pprint(args.__dict__)

    assert args.num_processes > 0, args.num_processes

    use_cuda = args.yes_cuda > 0 and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print('CUDA device_count {}'.format(torch.cuda.device_count())
          if use_cuda else 'CPU')
    if use_cuda and torch.cuda.device_count() > 0:
        print(torch.cuda.current_device(),
              torch.cuda.get_device_name(torch.cuda.current_device()))

    torch.manual_seed(args.seed)

    mp.set_start_method('spawn')  # mp

    with open(args.data_path, 'rb') as f:
        snli_dataset = pickle.load(f)

    train_loader, dev_loader, test_loader = \
        snli_dataset.get_dataloaders(batch_size=args.batch_size,
                                     num_workers=args.num_workers,
                                     pin_memory=use_cuda)
    print('#examples:',
          '#train', len(train_loader.dataset),
          '#dev', len(dev_loader.dataset),
          '#test', len(test_loader.dataset))

    model = MatchLSTM(args, snli_dataset.word2vec).to(device)
    model.share_memory()  # mp

    loss_func = nn.CrossEntropyLoss().to(device)

    processes = list()
    for i in range(args.num_processes):
        p = mp.Process(target=train_mp,
                       args=(i, device, snli_dataset, model, loss_func,
                             args))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    ckpt_path = os.path.join('./ckpt', '{}.pth'.format(args.name))
    if not os.path.exists(ckpt_path):
        print('Not found ckpt', ckpt_path)
        return

    # Load the best checkpoint
    load_model(model, ckpt_path)

    # Test
    evaluate_epoch(device, test_loader, model, args.epochs, loss_func, 'Test')


if __name__ == '__main__':
    main()
