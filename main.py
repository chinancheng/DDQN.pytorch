from ple import PLE
import os 
import cv2
import random
from ple.games.flappybird import FlappyBird
from agent import Agent
import numpy as np
from collections import deque
from utils import make_anim

os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ["SDL_VIDEODRIVER"] = "dummy"

game = FlappyBird()
p = PLE(game, fps=30, display_screen=False, force_fps=False)
p.init()
ACTION_SET = p.getActionSet()
SIZE_BUFFER = 50000
NUM_EPISODE = 200000
UPDATE_TARGET_FREQUENCY = 5
UPDATE_EPLISON_FREQUENCY = 100
SAVE_MODEL_FREQUENCY = 1000
SAVE_FILM_FREQUENCY = 500
BATCH_SIZE = 32
INITIAL_EPISODE = 100
agent = Agent(ACTION_SET)

reply_buffer = deque()
state = []
reward_logs = []
loss_logs = []
obs = p.getScreenGrayscale()
obs = cv2.resize(obs, (80, 80))
for _ in range(4):
    state.append(obs)
initial_state = np.stack([state], axis=0)

for episode in range(NUM_EPISODE):
    t = 0
    total_reward = 0
    p.reset_game()
    state = initial_state
    if episode % SAVE_FILM_FREQUENCY == 0 and episode > INITIAL_EPISODE: 
        agent.stop_eplison()
        frames = [p.getScreenRGB()] 
    while not p.game_over():
        action = agent.take_action(state)
        reward = p.act(ACTION_SET[action])
        if reward == 0:
            reward = 0.1
        elif reward == 1:
            print('FLY through the pipe')
        obs = p.getScreenGrayscale()
        if episode % SAVE_FILM_FREQUENCY == 0 and episode > INITIAL_EPISODE: 
            frames.append(p.getScreenRGB()) 
        obs = cv2.resize(obs, (80, 80))
        obs = np.reshape(obs, [1, 1, obs.shape[0], obs.shape[1]])
        state_new = np.append(state[:, 1:,...], obs, axis=1)
        action_matrix = np.zeros(len(ACTION_SET))
        action_matrix[action] = 1
        t += 1
        total_reward += reward
        if len(reply_buffer) > SIZE_BUFFER:
            reply_buffer.popleft()
        reply_buffer.append((state, action_matrix, reward, state_new, p.game_over()))
        state = state_new
    #print('Episode: {} t: {} Reward: {}' .format(episode, t, total_reward))
    if episode > INITIAL_EPISODE:
        if episode % SAVE_MODEL_FREQUENCY == 0:
            agent.save_model(episode, total_reward)
            np.save(os.path.join('./logs', 'loss.npy'), np.array(loss_logs))
            np.save(os.path.join('./logs', 'reward.npy'), np.array(reward_logs))
        if episode % SAVE_FILM_FREQUENCY == 0:  
            clip = make_anim(frames, fps=60, true_image=True).rotate(-90)
            clip.write_videofile("./movie/pg_{}.mp4".format(episode), fps=60)
            agent.restore_eplison()
        if episode % UPDATE_TARGET_FREQUENCY == 0:
            agent.update_target_network()
        batch = random.sample(reply_buffer, BATCH_SIZE)
        batch_state, batch_action, batch_reward, batch_state_new, batch_over = [], [], [], [], []
        for b in batch:
            batch_state.append(b[0][0])
            batch_action.append(b[1])
            batch_reward.append(b[2])
            batch_state_new.append(b[3][0])
            batch_over.append(float(not b[4]))
        batch_state = np.stack(batch_state)
        batch_action = np.stack(batch_action)
        batch_reward = np.stack(batch_reward)
        batch_state_new = np.stack(batch_state_new)
        batch_over = np.stack(batch_over)
        loss = agent.update_Q_network(batch_state, batch_action, batch_reward, batch_state_new, batch_over)
        loss_logs.extend([[episode, loss]]) 
        reward_logs.extend([[episode, total_reward]]) 
        print('Episode: {} t: {} Reward: {} Loss: {}' .format(episode, t, total_reward, loss))
        if episode % UPDATE_EPLISON_FREQUENCY == 0:
            agent.update_eplison()

    
     
