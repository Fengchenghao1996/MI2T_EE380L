import torch
from torch import nn
import torchvision.datasets as dsets
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class audioDataset(Dataset):

    def __init__(self, root_dir):
     
        self.root_dir = root_dir

        self.labels = ['cel','cla','flu','gac','gel','org','pia','sax','tru','vio','voi']
        self.videos = []
        self.numbers = []

     
        self.info = []
        num = 0
        for folder in self.labels:
            pth = os.path.join(self.root_dir,folder,'*.png')
            files = glob.glob(pth)
            print(files)
            self.videos = self.videos+files
            self.numbers.append(len(files))

        self.len = len(self.videos)
        self.pos = [sum(self.numbers[:i+1]) for i in range(11)]


    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        im =  mpimg.imread(self.videos[idx])

        for i in range(11):
            if idx < self.pos[i]:
                lab = i
                break
        return im,lab

#s = audioDataset('.\\input\\')
#s[2]
