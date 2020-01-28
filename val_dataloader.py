import pandas as pd
import torch
import torch.utils.data as data

import matplotlib.pyplot as plt

'''
filepath = 'E:/data/MNIST/test.csv'
'''

def dataloader(filepath):
    data = pd.read_csv(filepath)
    feature = data.to_numpy()/255.0
    return feature

class myDataset(data.Dataset):
    def __init__(self, feature):
        self.feature = feature

    def __getitem__(self, ind):
        feature = self.feature[ind]
        return feature.reshape(28,28)

    def __len__(self):
        return self.feature.shape[0]
    
if __name__ == "__main__":
    filepath = 'E:/data/MNIST/test.csv'
    feature = dataloader(filepath)
    plt.imshow(feature[0].reshape(28,28))
    plt.show()
