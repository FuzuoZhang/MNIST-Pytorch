import argparse

import os
import torch
import torch.utils.data
from dataloader import dataloader, myDataset

parser = argparse.ArgumentParser(description='CNN with MNIST')
parser.add_argument('--epoch', type=int, default=20, help='number of epochs to train')
parser.add_argument('--datapath', default='F:/data/MNIST/train.csv', help='training data path')
parser.add_argument('--train_bsize', type=int, default=4, help='training batch size')
parser.add_argument('--val_bsize', type=int, default=1, help='validation batch size')
parser.add_argument('--save_path', default='./results', help='save path')
parser.add_argument('--resume', type=str, default=None, help='resume path')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
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

    
