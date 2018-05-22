from __future__ import print_function
from __future__ import division
import argparse
import os
import time

import torch
import torch.utils.data
import torch.optim
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

from net import sphere20
from dataset import ImageList
import lfw_eval
import layer

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CosFace')

# DATA
parser.add_argument('--root_path', default='', type=str,
                    help='path to root path of images')
parser.add_argument('--train_list', default='~/dataset/CASIA-WebFace/CASIA-WebFace-112X96.txt', type=str,
                    help='path to training list')
parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                    help='input batch size for training (default: 512)')
# Model
parser.add_argument('--num_class', default=10572, type=int,
                    help='number of people(class) (default: 10572)')
# LR policy
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--step_size', type=list, default=[16000, 24000], metavar='SS',
                    help='lr decay step (default: [10,15,18])')  # [15000, 22000, 26000]
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=5e-4,
                    metavar='W', help='weight decay (default: 0.0005)')

parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_path', default='checkpoint/', type=str, metavar='PATH',
                    help='path to save checkpoint')
parser.add_argument('--no-cuda', type=bool, default=False,
                    help='disables CUDA training')
parser.add_argument('--workers', type=int, default=4, metavar='N',
                    help='how many workers to load data')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


def main():
    # --------------------------------------model----------------------------------------
    model = sphere20()
    model = torch.nn.DataParallel(model).cuda()
    print(model)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    model.module.save(args.save_path + 'CosFace_0_checkpoint.pth')

    # ------------------------------------load image---------------------------------------
    train_loader = torch.utils.data.DataLoader(
        ImageList(root=args.root_path, fileList=args.train_list,
                  transform=transforms.Compose([
                      transforms.RandomHorizontalFlip(),
                      transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
                      transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
                  ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    print('length of train Dataset: ' + str(len(train_loader.dataset)))
    print('Number of Classses: ' + str(args.num_class))

    # --------------------------------loss function and optimizer-----------------------------
    MCP = layer.MarginCosineProduct(512, args.num_class).cuda()
    # MCP = layer.AngleLinear(512, args.num_class).cuda()
    # MCP = torch.nn.Linear(512, args.num_class, bias=False).cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': MCP.parameters()}],
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # ----------------------------------------train----------------------------------------
    # lfw_eval.eval(args.save_path + 'CosFace_0_checkpoint.pth')
    for epoch in range(1, args.epochs + 1):
        # scheduler.step()
        train(train_loader, model, MCP, criterion, optimizer, epoch)
        model.module.save(args.save_path + 'CosFace_' + str(epoch) + '_checkpoint.pth')
        lfw_eval.eval(args.save_path + 'CosFace_' + str(epoch) + '_checkpoint.pth')
    print('Finished Training')


def train(train_loader, model, MCP, criterion, optimizer, epoch):
    model.train()
    print_with_time('Epoch {} start training'.format(epoch))
    time_curr = time.time()
    loss_display = 0.0

    for batch_idx, (data, target) in enumerate(train_loader, 1):
        iteration = (epoch - 1) * len(train_loader) + batch_idx
        adjust_learning_rate(optimizer, iteration, args.step_size)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        # compute output
        output = model(data)
        output = MCP(output, target)
        loss = criterion(output, target)
        loss_display += loss.data[0]
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            time_used = time.time() - time_curr
            loss_display /= args.log_interval
            INFO = ' Margin: {:.4f}, Scale: {:.2f}'.format(MCP.m, MCP.s)
            # INFO = ' lambda: {:.4f}'.format(MCP.lamb)
            print_with_time(
                'Train Epoch: {} [{}/{} ({:.0f}%)]{}, Loss: {:.6f}, Elapsed time: {:.4f}s({} iters)'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                    iteration, loss_display, time_used, args.log_interval) + INFO
            )
            time_curr = time.time()
            loss_display = 0.0


def print_with_time(string):
    print(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()) + string)


def adjust_learning_rate(optimizer, iteration, step_size):
    """Sets the learning rate to the initial LR decayed by 10 each step size"""
    if iteration in step_size:
        lr = args.lr * (0.1 ** (step_size.index(iteration) + 1))
        print_with_time('Adjust learning rate to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        pass


if __name__ == '__main__':
    main()
