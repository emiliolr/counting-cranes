import sys
import os
import json
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import warnings
import argparse
import time

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='PyTorch ASPDNet')

# parser.add_argument('train_json', metavar='TRAIN',
#                     help='path to train json')
# parser.add_argument('test_json', metavar='TEST',
#                     help='path to test json')

parser.add_argument('colab', metavar = 'COLAB', type = str2bool, help = 'are you using colab?')
parser.add_argument('config_fp', metavar = 'CONFIG', help = 'fp to config JSON file')

args = parser.parse_args()
config = json.load(open(args.config_fp, 'r'))
if args.colab:
    CODE_FP = config['code_filepath_colab']
else:
    CODE_FP = config['code_filepath_local']

sys.path.append(os.path.join(CODE_FP))
sys.path.append(os.path.join(CODE_FP, 'density_estimation'))
sys.path.append(os.path.join(CODE_FP, 'density_estimation', 'ASPDNet'))
# print(sys.path)

from ASPDNet.model import ASPDNet
from bird_dataset import *

# parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None, type=str,
#                     help='path to the pretrained model')

# parser.add_argument('gpu', metavar='GPU', type=str,
#                     help='GPU id to use.')

# parser.add_argument('task', metavar='TASK', type=str,
#                     help='task id to use.')

def main():
    global args, best_prec1

    best_prec1 = 1e6

    args = parser.parse_args()
    config = json.load(open(args.config_fp, 'r'))
    HYPERPARAMETERS = config['ASPDNet_params']
    args.tile_size = tuple(config['tile_size'])
    args.original_lr = HYPERPARAMETERS['learning_rate']
    args.lr = HYPERPARAMETERS['learning_rate']
    args.batch_size = HYPERPARAMETERS['batch_size']
    args.momentum = 0.95
    args.decay = 5 * 1e-4
    args.start_epoch = 0
    args.epochs = 2 #TODO; change this
    args.steps = [-1, 1, 100, 150]
    args.scales = [1, 1, 1, 1]
    # args.workers = 4
    args.seed = 1693
    args.print_freq = 4
    args.DATA_FP = config['data_filepath_local']
    # args.train_json = '../building_train.json'
    # args.test_json = '../building_test.json'
    # with open(args.train_json, 'r') as outfile:
    #     train_list = json.load(outfile)
    # with open(args.test_json, 'r') as outfile:
    #     val_list = json.load(outfile)

    # args.gpu = '0'
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.manual_seed(args.seed)

    model = ASPDNet()
    model = model.to(args.device)
    criterion = nn.MSELoss(size_average=False).to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.decay)

    indices = [14, 24, 7, 1, 28, 30, 5, 17, 10, 27, 12, 22, 16, 20, 25, 23, 0, 21, 3, 6, 33, 15, 19, 9, 18, 2, 32, 11, 29, 13, 26, 31, 8, 4]

    # if args.pre:
    #     if os.path.isfile(args.pre):
    #         print("=> loading checkpoint '{}'".format(args.pre))
    #         checkpoint = torch.load(args.pre)
    #         args.start_epoch = checkpoint['epoch']
    #         best_prec1 = checkpoint['best_prec1']
    #         model.load_state_dict(checkpoint['state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         print("=> loaded checkpoint '{}' (epoch {})"
    #               .format(args.pre, checkpoint['epoch']))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(args.pre))

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        train(indices, model, criterion, optimizer, epoch)
        prec1 = validate(indices, model, criterion)
        # args.task = "gao_large-vehicle_"
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        print(' * best MAE {mae:.3f} '
              .format(mae=best_prec1))
        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'arch': args.pre,
        #     'state_dict': model.state_dict(),
        #     'best_prec1': best_prec1,
        #     'optimizer': optimizer.state_dict(),
        # }, is_best, args.task)


def train(indices, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # train_loader = torch.utils.data.DataLoader(
    #     dataset.listDataset(train_list,
    #                         shuffle=True,
    #                         transform=transforms.Compose([
    #                             transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                                                         std=[0.229, 0.224, 0.225]),
    #                         ]),
    #                         train=True,
    #                         seen=model.seen,
    #                         batch_size=args.batch_size,
    #                         num_workers=args.workers),
    #     batch_size=args.batch_size)

    bird_dataset_train = BirdDataset(root_dir = args.DATA_FP,
                                     transforms = get_transforms('density_estimation', True),
                                     tiling_method = 'random',
                                     annotation_mode = 'points',
                                     num_tiles = 5,
                                     max_neg_examples = 1,
                                     tile_size = args.tile_size)
    dataset_train = torch.utils.data.Subset(bird_dataset_train, indices[ : 2]) #TODO: change this back!
    dataloader_train = DataLoader(dataset_train,
                                  batch_size = args.batch_size,
                                  shuffle = True,
                                  collate_fn = collate_tiles_density)

    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(dataloader_train.dataset), args.lr))

    model.train()
    end = time.time()

    for i, (imgs, targets, _) in enumerate(dataloader_train):
        data_time.update(time.time() - end)

        #TODO: verify this stuff (squeezing and unsqueezing?)
        for img, target in zip(imgs, targets):
            img = img.unsqueeze(0).to(args.device)
            img = Variable(img)
            output = model(img).squeeze()

            target = target.to(args.device)
            target = Variable(target)

            loss = criterion(output, target)

            losses.update(loss.item(), img.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                .format(
                epoch, i, len(dataloader_train), batch_time=batch_time,
                data_time=data_time, loss=losses))


def validate(indices, model, criterion):
    print ('begin test')
    # test_loader = torch.utils.data.DataLoader(
    #     dataset.listDataset(val_list,
    #                         shuffle=False,
    #                         transform=transforms.Compose([
    #                             transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                                                         std=[0.229, 0.224, 0.225]),
    #                         ]), train=False),
    #     batch_size=args.batch_size)
    bird_dataset_eval = BirdDataset(root_dir = args.DATA_FP,
                                    transforms = get_transforms('density_estimation', train = False),
                                    tiling_method = 'w_o_overlap',
                                    annotation_mode = 'points',
                                    tile_size = args.tile_size)
    dataset_val = torch.utils.data.Subset(bird_dataset_eval, indices[24 : 26]) #TODO: change this back!
    dataloader_val = DataLoader(dataset_val,
                                batch_size = args.batch_size,
                                shuffle = True,
                                collate_fn = collate_tiles_density)

    model.eval()

    mae = 0

    for i, (imgs, targets, counts) in enumerate(dataloader_val):
        for img, target, count in zip(imgs, targets, counts):
            img = img.unsqueeze(0).to(args.device)
            img = Variable(img)
            output = model(img)

            mae += abs(int(output.data.sum()) - count)

    mae = mae / len(dataloader_val)
    print(' * MAE {mae:.3f} '
          .format(mae=mae))

    return mae


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    args.lr = args.original_lr

    for i in range(len(args.steps)):

        scale = args.scales[i] if i < len(args.scales) else 1

        if epoch >= args.steps[i]:
            args.lr = args.lr * scale
            if epoch == args.steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
