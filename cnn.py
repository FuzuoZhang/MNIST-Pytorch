import torch
import torch.nn as nn
import torch.nn.functional as F

'''
典型用于手写数字识别的卷积神经网络
参考：周志华《机器学习》 P114
两个[卷积层+Relu+Max pooling]，三个全连接层
'''

class CNN(nn.Moudles):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1,6,3,stride=1,padding=1)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(6,16,5)
        self.pool2 = nn.MaxPool2d(2)

        self.lin1 = nn.Linear(16*5*5, 120)
        self.lin2 = nn.Linear(120,84)
        self.lin3 = nn.Linear(84, 10)


    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
    
        return F.log_softmax(x, dim=1)
