import pandas as pd
import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt 


'''
filepath = 'F:/data/MNIST/train.csv'
'''

def dataloader(filepath, random_state=None):
    data = pd.read_csv(filepath)

    label = data.label.values
    feature = data.loc[:,data.columns!='label'].values/255.0

    train_feture, val_feature, train_label, val_label = train_test_split(
        feature, label, test_size = 0.15, random_state=random_state
    )

    return train_feture, val_feature, train_label, val_label


class myDataset(data.Dataset):
    def __init__(self, feature, label):
        self.feature = feature
        self.label = label

    def __getitem__(self, ind):
        feature = self.feature[ind]
        label = self.label[ind]

        return feature.reshape(28,28), label

    def __len__(self):
        return len(self.label)


if __name__ == "__main__":
    filepath = 'F:/data/MNIST/train.csv'
    train_feture, val_feature, train_label, val_label = dataloader(filepath)
    print(train_feture.shape)
    print(train_label.shape)
    print(val_feature.shape)
    print(val_label.shape)
    i = 1
    print(train_label[i])
    plt.imshow(train_feture[i].reshape(28,28))
    plt.show()
    



