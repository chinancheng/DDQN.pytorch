import os 
import random
import numpy as np
from ple import PLE
from ple.games.flappybird import FlappyBird
from config import Config
from reply_buffer import Reply_Buffer
from agent import Agent
from utils import *

logs_path = './logs'
video_path = './video'

# Initial PLE environment
os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ["SDL_VIDEODRIVER"] = "dummy"
env = PLE(FlappyBird(), fps=30, display_screen=False, force_fps=False)
env.init()

reply_buffer = Reply_Buffer(Config.reply_buffer_size)
action_set = env.getActionSet()
agent = Agent(action_set)

reward_logs = []
loss_logs = []
estimate_logs = []

obs = resize(env.getScreenGrayscale())
initial_state = np.stack([[obs for _ in range(4)]], axis=0)

for episode in range(1, Config.total_episode+1):
    
    # reset env
    t = 0
    total_reward = 0
    total_estimate = 0
    env.reset_game()
    state = initial_state

    if episode % Config.save_film_frequency == 0 and episode > Config.initial_observe_episode: 
        agent.stop_epsilon()
        frames = [env.getScreenRGB()] 
    
    while not env.game_over():
        action, estimate = agent.take_action(state)
        reward = env.act(action_set[action])
        # soft reward for alive (0.1)
        if reward == 0:
            reward = 0.1
        elif reward == 1:
            print('FLY through the pipe')
        if episode % Config.save_film_frequency == 0 and episode > Config.initial_observe_episode: 
            frames.append(env.getScreenRGB()) 
        obs = resize(env.getScreenGrayscale())
        obs = np.reshape(obs, [1, 1, obs.shape[0], obs.shape[1]])
        state_new = np.append(state[:, 1:,...], obs, axis=1)
        action_onehot = np.zeros(len(action_set))
        action_onehot[action] = 1
        t += 1
        total_reward += reward
        total_estimate += estimate
        reply_buffer.append((state, action_onehot, reward, state_new, env.game_over()))
        state = state_new
    if episode > Config.initial_observe_episode:
        if episode % Config.save_model_frequency == 0:
            agent.save_model(episode, total_reward, logs_path)
            np.save(os.path.join(logs_path, 'loss.npy'), np.array(loss_logs))
            np.save(os.path.join(logs_path, 'reward.npy'), np.array(reward_logs))
            np.save(os.path.join(logs_path, 'estimate.npy'), np.array(estimate_logs))
        
        if episode % Config.save_film_frequency == 0:  
            os.makedirs(video_path, exist_ok=True)
            clip = make_anim(frames, fps=60, true_image=True).rotate(-90)
            clip.write_videofile(os.path.join(video_path, 'env_{}.mp4'.format(episode)), fps=60)
            agent.restore_epsilon()
        
        if episode % Config.update_target_frequency == 0:
            print('=> Update target network')
            agent.update_target_network()
        
        batch_state, batch_action, batch_reward, batch_state_new, batch_over = reply_buffer.sample(Config.batch_size)
        loss = agent.update_Q_network(batch_state, batch_action, batch_reward, batch_state_new, batch_over)
        
        loss_logs.extend([[episode, loss]]) 
        estimate_logs.extend([[episode, total_estimate]]) 
        reward_logs.extend([[episode, total_reward]]) 
        
        print('Episode: {} t: {} Reward: {:.3f} Loss: {:.3f} Estimate {:.3f}' .format(episode, t, total_reward, loss, total_estimate))
        
        if episode % Config.update_epsilon_frequency == 0:
            print('=> Update epsilon')
            agent.update_epsilon()

    
     
