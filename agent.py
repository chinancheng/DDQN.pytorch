import torch
from torch.nn.functional import mse_loss
from torch.autograd import Variable
import torch.optim as optim
import random
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
        # copy current_network to target network
        self.target_network.load_state_dict(self.Q_network.state_dict())
    
    def update_Q_network(self, state, action, reward, state_new, terminal):
        state = torch.from_numpy(state).float()/255.0
        action = torch.from_numpy(action).float()
        state_new = torch.from_numpy(state_new).float()/255.0
        terminal = torch.from_numpy(terminal).float()
        reward = torch.from_numpy(reward).float()
        state = Variable(state).cuda()
        action = Variable(action).cuda()
        state_new = Variable(state_new).cuda()
        terminal = Variable(terminal).cuda()
        reward = Variable(reward).cuda()
        self.Q_network.eval()
        self.target_network.eval()
        
        # use current network to evaluate action argmax_a' Q_current(s', a')_
        action_new = self.Q_network.forward(state_new).max(dim=1)[1].cpu().data.view(-1, 1)
        action_new_onehot = torch.zeros(Config.batch_size, self.action_number)
        action_new_onehot = Variable(action_new_onehot.scatter_(1, action_new, 1.0)).cuda()
        
        # use target network to evaluate value y = r + discount_factor * Q_tar(s', a')
        y = (reward + torch.mul(((self.target_network.forward(state_new)*action_new_onehot).sum(dim=1)*terminal), Config.discount_factor))
        
        # regression Q(s, a) -> y
        self.Q_network.train()
        Q = (self.Q_network.forward(state)*action).sum(dim=1)
        loss = mse_loss(input=Q, target=y.detach())
        
        # backward optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.data[0]

    def take_action(self, state):
        state = torch.from_numpy(state).float()/255.0
        state = Variable(state).cuda()
        
        self.Q_network.eval()
        estimate = self.Q_network.forward(state).max(dim=1)
        
        # with epsilon prob to choose random action else choose argmax Q estimate action
        if random.random() < self.epsilon:
            return random.randint(0, self.action_number-1)
        else:
            return estimate[1].data[0]
    
    def update_epsilon(self):
        if self.epsilon > Config.min_epsilon:
            self.epsilon -= Config.epsilon_discount_rate
    
    def stop_epsilon(self):
        self.epsilon_tmp = self.epsilon        
        self.epsilon = 0        
    
    def restore_epsilon(self):
        self.epsilon = self.epsilon_tmp        
    
    def save(self, episode, reward, logs_path):
        # Store best reward model
        if reward > self.best_reward:
            os.makedirs(logs_path, exist_ok=True)
            for li in glob.glob(os.path.join(logs_path, '*.pth')):
                os.remove(li)
            logs_path = os.path.join(logs_path, 'model-{}.pth' .format(episode))
            self.Q_network.save(logs_path, step=episode, optimizer=self.optimizer)
            self.best_reward = reward
            print('=> Save {}' .format(logs_path)) 

    def restore(self, logs_path):
        episode = self.Q_network.load(logs_path)
        _ = self.target_network.load(logs_path)
        print('=> Restore {}' .format(logs_path)) 
        
        return episode

        
    

