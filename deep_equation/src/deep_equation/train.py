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

from model import CNN
from loader import DataLoader


# ##### ============================ DATASET LOADER ============================ #####

# class SquarePad:
#     def __call__(self, image):
#         max_wh = max(image.size)
#         p_left, p_top = [(max_wh - s) // 2 for s in image.size]
#         p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
#         padding = (p_left, p_top, p_right, p_bottom)
#         return F.pad(image, padding, 0, 'constant')


# target_image_size = (32, 32)
# transformed=transforms.Compose([
#     SquarePad(),
#     transforms.Resize(target_image_size),
#     transforms.ToTensor(),
# #     transforms.RandomInvert(p=0.5),
# ])


# train_data = datasets.MNIST(
#     root = 'data',
#     train = True,                         
#     transform = transformed, 
#     download = True,            
# )
# test_data = datasets.MNIST(
#     root = 'data', 
#     train = False, 
#     transform = transformed
# )

# print(train_data.data.size())
# print(train_data.targets.size())


# BATCH_SIZE = 100
# loaders = {
#     'train' : torch.utils.data.DataLoader(train_data, 
#                                           batch_size=BATCH_SIZE, 
#                                           shuffle=True, 
#                                           num_workers=1),
    
#     'test'  : torch.utils.data.DataLoader(test_data, 
#                                           batch_size=BATCH_SIZE, 
#                                           shuffle=True, 
#                                           num_workers=1),
# }


# ### ========= VISUALIZE SAMPLES ========= ###

# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt


# figure = plt.figure(figsize=(10, 8))
# cols, rows = 5, 5
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(train_data), size=(1,)).item()
#     img, label = train_data[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     plt.title(label)
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="gray")
# figure.savefig("deep_equation/input/input_examples.png")


symbols = ['+', '-', '*', '/']
d_op = {'+': [1, 0, 0, 0], '-': [0, 1, 0, 0], '*': [0, 0, 1, 0], '/': [0, 0, 0, 1]}


L = []
for i in range(10):
    for j in range(1,10):
        som = i+j
        sub = i-j
        div = round(i/j, 2)
        mult = i*j
        if som not in L:
            L.append(som)
        if sub not in L:
            L.append(sub)
        if mult not in L:
            L.append(mult)
        if div not in L:
            L.append(div)
            
labels_dict = {value: key for (key, value) in enumerate(sorted(L))}
labels_dict[-1000] = len(labels_dict)

inv_labels_dict = {v: k for k, v in labels_dict.items()}
print(labels_dict)
print('\n', len(labels_dict))


def calculate_tensor_operation(X, y, op, batch_size=100):
    ''' Concatenate tensors channelwise
    '''
    T = torch.empty(size=(batch_size, 3, 32, 32))
    L = torch.empty(size=(batch_size,))
    for i in range(batch_size):
        rand1, rand2 = random.sample(range(0, batch_size), 2)

        T1, L1 = X[rand1], y[rand1]
        T2, L2 = X[rand2], y[rand2]
        
        v = d_op[op]
        if op == '+':
            res = L1 + L2
        elif op == '-':
            res = L1 - L2
        elif op == '*':
            res = L1 * L2
        else:
            res = L1 / L2
            if L2 == 0:
                res = torch.tensor(-1000)
                
        item = round(res.item(), 2)
        value = labels_dict[item]
        V = torch.tensor([32*[int(32/4)*v]])
        t = torch.cat((T1, T2, V), 0)
        
        T[i] = t
        L[i] = value
        
    L = L.type(torch.LongTensor)
    return T, L


class Train:
    def __init__(self):
        ''' 
        Initialize loader parameters
        '''
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(cnn.parameters(), lr = 0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=4, gamma=0.05)
        self.num_epochs = 1
        
        
    def calculate_tensor_operation(self, X, y, op, batch_size=100):
        ''' 
        Concatenate tensors channelwise
        '''
        T = torch.empty(size=(batch_size, 3, 32, 32))
        L = torch.empty(size=(batch_size,))
        for i in range(batch_size):
            rand1, rand2 = random.sample(range(0, batch_size), 2)

            T1, L1 = X[rand1], y[rand1]
            T2, L2 = X[rand2], y[rand2]

            v = d_op[op]
            if op == '+':
                res = L1 + L2
            elif op == '-':
                res = L1 - L2
            elif op == '*':
                res = L1 * L2
            else:
                res = L1 / L2
                if L2 == 0:
                    res = torch.tensor(-1000)

            item = round(res.item(), 2)
            value = labels_dict[item]
            V = torch.tensor([32*[int(32/4)*v]])
            t = torch.cat((T1, T2, V), 0)

            T[i] = t
            L[i] = value

        L = L.type(torch.LongTensor)
        return T, L


    def train(self, cnn, loaders):
        ''' 
        Training module
        '''
        cnn.train()
        
        # Train the model
        total_step = len(loaders['train'])

        for epoch in range(self.num_epochs):
            for i, (images, labels) in enumerate(loaders['train']):

                # gives batch data, normalize x when iterate train_loader
                X = Variable(images)   # batch x
                y = Variable(labels)   # batch y

                for op in symbols:

                    b_x, b_y = self.calculate_tensor_operation(X, y, op)

                    output = cnn(b_x)[0]
                    loss = self.loss_func(output, b_y)

                    # clear gradients for this training step   
                    self.optimizer.zero_grad()           

                    # backpropagation, compute gradients 
                    loss.backward()    
                    # apply gradients             
                    self.optimizer.step()                

                    if (i+1) % 100 == 0:
                        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                               .format(epoch + 1, self.num_epochs, i + 1, total_step, loss.item()))

            self.scheduler.step()

        pass
        
if __name__ == '__main__':
    
    D = DataLoader()
    D.generate_examples()
    loaders = D.load_datasets()
    cnn = CNN()

    t = Train()
    t.train(cnn, loaders)
    torch.save(cnn.state_dict(), 'deep_equation/input/model.pth')


# def train(num_epochs, cnn, sched, loaders):
    
#     cnn.train()
        
#     # Train the model
#     total_step = len(loaders['train'])
        
#     for epoch in range(num_epochs):
#         for i, (images, labels) in enumerate(loaders['train']):
            
#             # gives batch data, normalize x when iterate train_loader
#             X = Variable(images)   # batch x
#             y = Variable(labels)   # batch y
            
#             for op in symbols:
                
#                 b_x, b_y = calculate_tensor_operation(X, y, op)
                
#                 output = cnn(b_x)[0]
#                 loss = loss_func(output, b_y)

#                 # clear gradients for this training step   
#                 optimizer.zero_grad()           

#                 # backpropagation, compute gradients 
#                 loss.backward()    
#                 # apply gradients             
#                 optimizer.step()                
        
#                 if (i+1) % 100 == 0:
#                     print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
#                            .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
#                     pass
        
#         sched.step()
#         pass
    
#     pass
