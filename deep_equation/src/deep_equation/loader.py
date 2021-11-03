import math
import torch
import random

from torch import optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.nn import functional as F
import torch.nn as nn

import torchvision
import torchvision.transforms.functional as F
from torchvision import datasets, models, transforms
from torchvision import datasets
from torchvision.transforms import ToTensor


class DataLoader:
    def __init__(self):
        ''' 
        Initialize loader parameters
        '''
        self.BATCH_SIZE = 100


    def load_mnist(self):
        '''
        Load MNIST data
        '''
        transformation=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])


        train_data = datasets.MNIST(
            root = 'data',
            train = True,                         
            transform = transformation, 
            download = True,            
        )
        test_data = datasets.MNIST(
            root = 'data', 
            train = False, 
            transform = transformation
        )
        
        print(train_data.data.size())
        print(train_data.targets.size())
        return train_data, test_data
    
        
    def load_datasets(self):
        '''
        Load train and test data, returning data in batches
        '''
        train_data, test_data = self.load_mnist()
        loaders = {
            'train' : torch.utils.data.DataLoader(train_data, 
                                                  batch_size=self.BATCH_SIZE, 
                                                  shuffle=True, 
                                                  num_workers=1),

            'test'  : torch.utils.data.DataLoader(test_data, 
                                                  batch_size=self.BATCH_SIZE, 
                                                  shuffle=True, 
                                                  num_workers=1),
        }
        return loaders


    def generate_examples(self):
        '''
        Generate examples of the input data
        '''
        train_data = self.load_mnist()[0]

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt


        figure = plt.figure(figsize=(10, 8))
        cols, rows = 5, 5
        for i in range(1, cols * rows + 1):
            sample_idx = torch.randint(len(train_data), size=(1,)).item()
            img, label = train_data[sample_idx]
            figure.add_subplot(rows, cols, i)
            plt.title(label)
            plt.axis("off")
            plt.imshow(img.squeeze(), cmap="gray")
        figure.savefig("deep_equation/input/input_examples.png")


