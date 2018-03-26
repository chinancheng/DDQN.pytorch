import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
from torch.autograd import Variable
import torch.optim as optim
import random
import numpy as np
import pdb

class Agent:
    def __init__(self, ACTION_SET, DISCOUNT_FACTOR=0.99, LEARNING_RATE=1e-5):
        self.ACTION_SET = ACTION_SET
        self.ACTION_NUM = len(ACTION_SET)
        self.DISCOUNT_FACTOR = DISCOUNT_FACTOR
        self.LEARNING_RATE = LEARNING_RATE
        self.build_network()

    def build_network(self):
        self.Q_network = Model(self.ACTION_NUM).cuda()
        self.target_network = self.Q_network
        self.optimizer = optim.Adam(self.Q_network.parameters(), lr=self.LEARNING_RATE)
    
    def update_target_network(self):
        self.target_network = self.Q_network
    
    def update_Q_network(self, state, reward, action):
        state = torch.from_numpy(state).float()
        state = Variable(state)
        y = reward + self.DISCOUNT_FACTOR*self.target_network.forwad(state).max(dim=1)
        Q = (self.Q_network.forward(state)*action).sum()
        loss = mse(input=Q, target=y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def take_action(self, state):
        state = torch.from_numpy(state).float()/255.0
        state = Variable(state).cuda()
        if random.random() > 0.99:
            return self.ACTION_SET[random.randint(0, self.ACTION_NUM)]
        else:
            return self.ACTION_SET[self.target_network.forward(state).max(dim=1)[1].data[0]]
        

class Model(nn.Module):
    
    def __init__(self, ACTION_NUM):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(8, 8), stride=4, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU())
        self.fc2 = nn.Linear(in_features=256, out_features=ACTION_NUM)
    
    def forward(self, observation):
        out1 = self.conv1(observation)        
        out2 = self.conv2(out1)        
        out3 = self.conv3(out2)
        out4 = self.fc1(out3.view(-1, 256))        
        out = self.fc2(out4)
        
        return out

            
            
        
