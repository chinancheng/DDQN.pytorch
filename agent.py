import torch.nn as nn
from torch.nn.functional import mse_loss


class Agent:
    def __init__(self, ACTION, DISCOUNT_FACTOR=0.99):
        self.ACTION = ACTION
        self.build_network()
        self.DISCOUNT_FACTOR = DISCOUNT_FACTOR
        self.LEARNING_RATE = 1e-5

    def build_network(self):
        self.Q_network = Model(self.ACTION)
        self.target_network = self.Q_network
        self.optimizer = optim.Adam(self.Q_network.parameters(), lr=self.LEARNING_RATE)
    
    def update_target_network(self):
        self.target_network = self.Q_network
    
    def update_Q_network(self, observation, reward, action):
        y = reward + self.DISCOUNT_FACTOR*self.target_network.forwad(observation).max(dim=1)
        Q = (self.Q_network.forward(observation)*action).sum()
        loss = mse(input=Q, target=y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class Model(nn.module):
    def __init__(self, ACTION):
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(8, 8), stride=4, padding=7),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1))
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=256, out_features=256)
            nn.ReLU())
        self.fc2 = nn.Linear(in_features=256, out_features=ACTION)
    
    def forward(self, observation):
        out1 = self.conv1(observation)        
        out2 = self.conv2(out1)        
        out3 = self.conv3(out2)        
        out4 = self.fc1(out3.view(-1))        
        out = self.fc2(out4)
        
        return out

            
            
        
