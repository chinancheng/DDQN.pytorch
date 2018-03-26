from ple import PLE
import os 
import cv2
from ple.games.pong import Pong
from agent import Agent
import numpy as np

os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ["SDL_VIDEODRIVER"] = "dummy"

game = Pong(width=80, height=80)
p = PLE(game, fps=30, display_screen=False, force_fps=False)
p.init()
ACTION_SET = p.getActionSet()
nb_frames = 1000
reward = 0.0
agent = Agent(ACTION_SET)

state = []
obs = p.getScreenRGB()
obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
ret, obs = cv2.threshold(obs, 1, 255, cv2.THRESH_BINARY)
for _ in range(4):
    state.append(obs)
state = np.stack([state], axis=0)
while True:
    if p.game_over(): #check if the game is over
        p.reset_game()
    action = agent.take_action(state)
    reward = p.act(action)
    obs = p.getScreenRGB()
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
    ret, obs = cv2.threshold(obs, 1, 255, cv2.THRESH_BINARY)
    obs = np.reshape(obs, [1, 1, obs.shape[0], obs.shape[1]])
    state_new = np.append(state[:, 1:,...], obs, axis=1)
    exit()
    #action = agent.take_action(observation)
    #reward = p.act(action)


    
     
