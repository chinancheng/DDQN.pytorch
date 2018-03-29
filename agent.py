import torch
from torch.nn.functional import mse_loss
from torch.autograd import Variable
import torch.optim as optim
import random
import numpy as np
import pdb
import glob
import os
from config import Config
from model import Model

class Agent:
    def __init__(self, action_set):
        self.action_set = action_set
        self.action_number = len(action_set)
        self.epsilon = Config.initial_epsilon
        self.best_reward = -100
        self.build_network()

    def build_network(self):
        self.Q_network = Model(self.action_number).cuda()
        self.target_network = Model(self.action_number).cuda()
        self.optimizer = optim.Adam(self.Q_network.parameters(), lr=Config.lr)
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.Q_network.state_dict())
    
    def update_Q_network(self, state, action, reward, state_new, terminal):
        state = torch.from_numpy(state).float()
        action = torch.from_numpy(action).float()
        state_new = torch.from_numpy(state_new).float()
        terminal = torch.from_numpy(terminal).float()
        reward = torch.from_numpy(reward).float()
        state = Variable(state).cuda()
        action = Variable(action).cuda()
        state_new = Variable(state_new).cuda()
        terminal = Variable(terminal).cuda()
        reward = Variable(reward).cuda()
        
        self.target_network.eval()
        y = (reward + torch.mul((self.target_network.forward(state_new).max(dim=1)[0]*terminal), Config.discount_factor))
        
        self.Q_network.train()
        Q = (self.Q_network.forward(state)*action).sum(dim=1)
        loss = mse_loss(input=Q, target=y.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.data[0]

    def take_action(self, state):
        state = torch.from_numpy(state).float()/255.0
        state = Variable(state).cuda()
        
        self.Q_network.eval()
        estimate = self.Q_network.forward(state).max(dim=1)
        
        if random.random() < self.epsilon:
            return random.randint(0, self.action_number-1), estimate[0].data[0]
        else:
            return estimate[1].data[0], estimate[0].data[0]
    
    def update_epsilon(self):
        if self.epsilon > Config.min_epsilon:
            self.epsilon = self.epsilon*Config.epsilon_discount_rate
    
    def stop_epsilon(self):
        self.epsilon_tmp = self.epsilon        
        self.epsilon = 0        
    
    def restore_epsilon(self):
        self.epsilon = self.epsilon_tmp        
    
    def save_model(self, episode, reward, logs_path):
        if reward > self.best_reward:
            os.makedirs(logs_path, exist_ok=True)
            for li in glob.glob(os.path.join(logs_path, '*.pth')):
                os.remove(li)
            model_path = os.path.join(logs_path, 'model-{}.pth' .format(episode))
            self.Q_network.save(model_path, step=episode, optimizer=self.optimizer)
            self.best_reward = reward 


