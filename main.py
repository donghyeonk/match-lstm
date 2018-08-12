import argparse
from datetime import datetime
import pickle
import pprint
import torch
import torch.nn.functional as F
import torch.optim as optim
from dataset import SNLIData
from model import MatchLSTM


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2018)
parser.add_argument('--data_path', type=str, default='./data/snli.pkl')
parser.add_argument('--num_classes', type=int, default=3)

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--grad_max_norm', type=float, default=0)

parser.add_argument('--input_size', type=int, default=300)
parser.add_argument('--embedding_dim', type=int, default=300)
parser.add_argument('--hidden_size', type=int, default=300)
parser.add_argument('--num_layers', type=int, default=1)

parser.add_argument('--batch_size', type=int, default=30)
parser.add_argument('--epochs', type=int, default=20)

parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--yes_cuda', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=4)


def train_epoch(device, loader, model, epoch, optimizer, config):
    model.train()
    train_loss = 0.
    example_count = 0
    correct = 0
    start_t = datetime.now()
    for batch_idx, ex in enumerate(loader):
        target = torch.tensor(ex[2], device=device)
        optimizer.zero_grad()
        output = model(ex[0], ex[1])
        loss = F.nll_loss(output, target)
        loss.backward()
        if config.grad_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           config.grad_max_norm)
        optimizer.step()

        batch_loss = len(output) * loss.item()
        train_loss += batch_loss
        example_count += len(target)

        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()

        if (batch_idx + 1) % config.log_interval == 0 \
                or batch_idx == len(loader) - 1:
            _progress = \
                '{} Train Epoch {}, [{}/{} ({:.1f}%)],\tBatch Loss: {:.6f}' \
                .format(datetime.now(), epoch,
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


def evaluate_epoch(device, loader, model, epoch, mode):
    model.eval()
    eval_loss = 0.
    correct = 0
    start_t = datetime.now()
    with torch.no_grad():
        for batch_idx, ex in enumerate(loader):
            target = torch.tensor(ex[2], device=device)
            output = model(ex[0].to(device), ex[1].to(device))
            loss = F.nll_loss(output, target)
            eval_loss += len(output) * loss.item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    eval_loss /= len(loader.dataset)
    acc = correct / len(loader.dataset)
    print('{} {} Epoch {}, Avg. Loss: {:.6f}, '
          'Accuracy: {}/{} ({:.1f}%)'.format(datetime.now()-start_t, mode,
                                             epoch, eval_loss,
                                             correct, len(loader.dataset),
                                             100. * acc))
    return eval_loss, acc


def main():
    args = parser.parse_args()
    pprint.PrettyPrinter().pprint(args.__dict__)

    use_cuda = args.yes_cuda > 0 and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    print('CUDA device_count {0}'.format(torch.cuda.device_count())
          if use_cuda else 'CPU')

    with open(args.data_path, 'rb') as f:
        snli_dataset = pickle.load(f)

    train_loader, valid_loader, test_loader = \
        snli_dataset.get_dataloaders(batch_size=args.batch_size,
                                     num_workers=args.num_workers,
                                     pin_memory=use_cuda)
    print(len(train_loader.dataset), len(valid_loader.dataset),
          len(test_loader.dataset))

    model = MatchLSTM(args, snli_dataset.word2vec).to(device)

    optimizer = optim.Adam(model.get_req_grad_params(), lr=args.lr,
                           betas=(0.9, 0.999))

    best_loss = float('inf')
    best_acc = 0.
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        train_epoch(device, train_loader, model, epoch, optimizer, args)

        valid_loss, valid_acc = \
            evaluate_epoch(device, valid_loader, model, epoch, 'Valid')
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_acc = valid_acc
            best_epoch = epoch
        print('\tLowest Valid Loss {:.6f}, Acc. {:.1f}%, Epoch {}'.
              format(best_loss, 100 * best_acc, best_epoch))

        evaluate_epoch(device, test_loader, model, epoch, 'Test')

        # TODO learning rate decay


if __name__ == '__main__':
    main()
