from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dset
import torch
import numpy as np
import os
from ipdb import set_trace as debug

def shrink_dataset(data, labels, keep_ratio=1.):
    size = data.shape[0]
    np.random.seed(size)
    inds = np.arange(size)
    np.random.shuffle(inds)
    inds_keep = inds[0:int(size*keep_ratio)]
    inds_keep.sort()
    data_keep = data[inds_keep]
    labels_keep = labels[inds_keep]
    return data_keep, labels_keep

class CIFAR10(Dataset):
    def __init__(self,normal_class,train=True, sup=True, keep_ratio=1.0):
        self.root = './datasets/cifar10'
        self.sup = sup
        self.data, self.targets = self.load_data_CIFAR10(normal_class,train,keep_ratio)

    def load_data_CIFAR10(self,normal_class, train,keep_ratio):
        dataset = dset.CIFAR10(self.root, train=train, download=True)
        data = np.array(dataset.data).astype('float32')
        data = self.norm(data)
        targets = np.array(dataset.targets)
        targets = (targets!=normal_class).astype('float32')
        if not self.sup:
            indexs = np.where(targets==0)[0]
            data = data[indexs]
            targets = targets[indexs]
        # data, targets = shrink_dataset(data,targets,keep_ratio)
        return data,targets

    def norm(self, data, mu=1):
        # return 2 * (data / 255.) - mu
        return data/255.

    def __getitem__(self,index):
        return self.data[index].transpose(2,0,1),self.targets[index]

    def __len__(self):
        return self.data.shape[0]


if __name__=='__main__':
    trainset = CIFAR10(normal_class=0,train=True,sup=False)
    debug()
