import torch.nn as nn


class CNN(nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=3,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),
#             nn.BatchNorm2d(16),
            nn.LeakyReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 1), 
#             nn.BatchNorm2d(32),
            nn.LeakyReLU(),                      
            nn.MaxPool2d(2),                
        )
        self.conv3 = nn.Sequential(         
            nn.Conv2d(32, 64, 4, 1, 1),
#             nn.BatchNorm2d(64),
            nn.LeakyReLU(),                      
            nn.MaxPool2d(1),                
        )
        self.conv4 = nn.Sequential(         
            nn.Conv2d(64, 128, 3, 1, 2),
#             nn.BatchNorm2d(128),
            nn.LeakyReLU(),                      
            nn.MaxPool2d(1),                
        )
        self.dropout = nn.Dropout(0.3)
        self.num_classes = 96
        self.dense_units = 224
        self.linear = nn.Linear(128 * 8 * 8, self.dense_units)
        self.last = nn.Linear(self.dense_units, self.num_classes)
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1) # flatten the output of conv2 to (batch_size, n_filters * height * width)
        x = self.dropout(x)
        x = self.linear(x)
        x = self.dropout(x)
        output = self.last(x)
        return output, x    # return x for visualization
