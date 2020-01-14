import argparse

import os
import time
import torch
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
from cnn import CNN
from logger import setup_logger
from dataloader import dataloader, myDataset

parser = argparse.ArgumentParser(description='CNN with MNIST')
parser.add_argument('--epoch', type=int, default=20, help='number of epochs to train')
parser.add_argument('--datapath', default='F:/data/MNIST/train.csv', help='training data path')
parser.add_argument('--train_bsize', type=int, default=4, help='training batch size')
parser.add_argument('--val_bsize', type=int, default=1, help='validation batch size')
parser.add_argument('--save_path', default='./results', help='save path')
parser.add_argument('--resume', type=str, default=None, help='resume path')
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--gpu', type='str', default='0', help='GPU ID')
args = parser.parse_args()



def main():
    global args
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    
    train_feture, val_feature, train_label, val_label = dataloader(args.datapath)
    TrainDataset = torch.utils.data.DataLoader(myDataset(train_feture, train_label),
                    batch_size = args.train_bsize, shuffle=False, num_workers=1, drop_last=False)
    ValDataset = torch.utils.data.DataLoader(myDataset(val_feature, val_label),
                    batch_size=args.val_bsize, shuffle=False, num_workers=1, drop_last=False)

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    log = setup_logger(args.save_path+'/training.log')
    for key,value in sorted(vars(args).items()):
        log.info(str(key)+":"+str(value))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN()
    if torch.cuda.device_count()>1:
        model = nn.DataParallel(model).cuda()
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    args.start_epoch = 0

    if args.resume:
        if os.path.isfile(args.resume):
            log.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            log.info("=> loaded checkpoint '{}'(epoch {})"
                    .format(args.resume, checkpoint['epoch']))
        else:
            log.info("=> no checkpoint found at '{}'".format(args.resume))
            log.info("=> will start from scratch")
    else:
        log.info("Not Resume")

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epoch):
        log.info('This is {}-th epoch'.format(epoch))
        pass


def train(dataset, model, device, optimizer, log, epoch=0):
    model.train()
    for batch_idx, (feature, label) in enumerate(dataset):
        feature, label = feature.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(feature)
        loss = F.nll_loss(output, label)
        loss.backward()
        optimizer.step()
        






