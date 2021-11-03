import torch
from torchvision import datasets, models, transforms

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
        
    
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
            transforms.RandomInvert(p=0.5),
            transforms.GaussianBlur(kernel_size=(9, 9), sigma=(0.1, 2.5)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.7, hue=0.5),
            transforms.RandomAdjustSharpness(sharpness_factor=0.1, p=0.5),
            transforms.RandomAutocontrast(p=0.5),
            transforms.RandomResizedCrop(size=(32, 32), scale=(0.7, 1.0)),
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


