import json
import torch
import random
from torch import optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn as nn

from model import CNN
from loader import DataLoader


symbols = ['+', '-', '*', '/']
d_op = {'+': [1, 0, 0, 0], '-': [0, 1, 0, 0], '*': [0, 0, 1, 0], '/': [0, 0, 0, 1]}
    
with open('deep_equation/src/deep_equation/labels_dict.json', 'r') as fp:
    mapping_dict = json.loads(fp.read())
    
    
class Train:
    def __init__(self):
        ''' 
        Initialize loader parameters
        '''
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(cnn.parameters(), lr = 0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)
        self.num_epochs = 20
        self.labels_dict = mapping_dict

        
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
                    res = torch.tensor(-99999999999999)

            item = round(res.item(), 2)
            value = self.labels_dict['{0:.2f}'.format(item)]
            V = torch.tensor([32*[int(32/4)*v]])
            t = torch.cat((T1, T2, V), 0)

            T[i] = t
            L[i] = value

        L = L.type(torch.LongTensor)
        return T, L


    def train(self, cnn, loaders):
        ''' 
        Training function
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
        
        
if __name__ == '__main__':
    D = DataLoader()
    D.generate_examples()
    loaders = D.load_datasets()
    cnn = CNN()

    t = Train()
    t.train(cnn, loaders)
    torch.save(cnn.state_dict(), 'deep_equation/input/model.pth')