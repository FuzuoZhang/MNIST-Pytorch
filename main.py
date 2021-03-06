import argparse

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
<<<<<<< HEAD
=======
import torch.nn as nn
>>>>>>> 437352c110228df91c51503d92636f758eccfe29
import torch.nn.functional as F
from cnn import CNN
from logger import setup_logger
import dataloader as D1
import val_dataloader as D2
#from dataloader import dataloader, myDataset

parser = argparse.ArgumentParser(description='CNN with MNIST')
parser.add_argument('--epoch', type=int, default=20, help='number of epochs to train')
parser.add_argument('--datapath', default='/home/zhangfz/data/MNIST/train.csv', help='training data path')
parser.add_argument('--train_bsize', type=int, default=20, help='training batch size')
parser.add_argument('--val_bsize', type=int, default=1, help='validation batch size')
parser.add_argument('--save_path', default='./results', help='save path')
parser.add_argument('--resume', type=str, default='./results/checkpoint.pth', help='resume path')
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--stepsize', type=int, default=1, help='step size for learning rate')
parser.add_argument('--gamma', type=float, default=0.7, help='gamma for lr schedular')
parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
<<<<<<< HEAD
=======
parser.add_argument('--testpath', type=str, default='/home/zhangfz/data/MNIST/test.csv', help='test data path')
>>>>>>> 437352c110228df91c51503d92636f758eccfe29
args = parser.parse_args()



def main():
    global args
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    
    train_feture, test_feature, train_label, test_label = D1.dataloader(args.datapath)
    TrainDataset = torch.utils.data.DataLoader(D1.myDataset(train_feture, train_label),
                    batch_size = args.train_bsize, shuffle=False, num_workers=1, drop_last=False)
    TestDataset = torch.utils.data.DataLoader(D1.myDataset(test_feature, test_label),
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
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)
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
        train(TrainDataset, model, device, optimizer, log, epoch)

<<<<<<< HEAD
        savefile = args.save_path + 'checkpoint.pth'
=======
        savefile = args.save_path + '/checkpoint.pth'
>>>>>>> 437352c110228df91c51503d92636f758eccfe29
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()},
            savefile)
        
        scheduler.step() # 更新学习率
<<<<<<< HEAD
    log.info("full training time = {: 2f} hours".format(
        (time.time()-start_time)/3600.0
    ))

=======
        test(TestDataset, model, device, log, epoch)
    log.info("full training time = {: 2f} hours".format(
        (time.time()-start_time)/3600.0
    ))
     
    val(model,device)
>>>>>>> 437352c110228df91c51503d92636f758eccfe29

    
def train(dataset, model, device, optimizer, log, epoch=0):
    model.train()
    for batch_idx, (feature, label) in enumerate(dataset):
<<<<<<< HEAD
        feature, label = feature.to(device), label.to(device)
=======
        feature, label = feature.float().to(device), label.to(device)
>>>>>>> 437352c110228df91c51503d92636f758eccfe29
        #把所有变量的grad成员值变为0
        optimizer.zero_grad()
        output = model(feature)
        #计算损失函数
        loss = F.nll_loss(output, label)
        loss.backward()
        #模型更新
        optimizer.step()
        if batch_idx % 500 == 0:
            log.info("Epoch: {}/{}   Training Loss: {:.3f}"
            .format(epoch+1, args.epoch, loss))

<<<<<<< HEAD
'''
def test(dataset, model, device, log):
    
    model.eval()
    for batch_idx, (feature, label) in enumerate(dataset):
        feature, label = feature.to(device), label.to(device)
'''
=======

def test(dataset, model, device, log, epoch=0):
    model.eval()
    test_loss = 0
    correct = 0 
    with torch.no_grad():
        for batch_idx, (feature, label) in enumerate(dataset):
            feature, label = feature.float().to(device), label.to(device)
            output = model(feature)
            test_loss = F.nll_loss(output, label, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()
    test_loss /= len(dataset.dataset)
    log.info("Epoch: {}/{}    Test Average loss: {:.3f}, Accuracy: {}/{} ({:.2f}%)"
            .format(epoch+1, args.epoch, test_loss,correct,len(dataset.dataset),
            100.*correct/len(dataset.dataset)))


def val(model, device):
    #load data 
    feature = D2.dataloader(args.testpath)
    ValDataset = torch.utils.data.DataLoader(D2.myDataset(feature))

    n = len(ValDataset)
    imgid = np.arange(n)+1
    label = np.zeros((n,),dtype=np.int8)

    model.eval()
    with torch.no_grad():
        for batch_idx, feature in enumerate(ValDataset):
            feature = feature.float().to(device)
            output = model(feature)
            pred = output.argmax(dim=1, keepdim=True)
            label[batch_idx] = pred

    df = pd.DataFrame({
        'ImageId':imgid,
        'Label': label
    })

    filename = args.save_path+'/submission.csv'
    df.to_csv(filename, index=False)
>>>>>>> 437352c110228df91c51503d92636f758eccfe29

if __name__ == "__main__":
    main()





