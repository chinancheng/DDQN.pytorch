import os 
import random
import numpy as np
import argparse
from ple import PLE
from ple.games.flappybird import FlappyBird
from config import Config
from reply_buffer import Reply_Buffer
from agent import Agent
from utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--logs_path', dest='logs_path', help='path of the checkpoint folder',
                        default='./logs', type=str)
    parser.add_argument('-v', '--video_path', dest='video_path', help='path of the video folder',
                        default='./video', type=str)
    parser.add_argument('-r', '--restore', dest='restore', help='restore checkpoint',
                        default=None, type=str)
    parser.add_argument('-t', '--train', dest='train', help='train policy or not',
                        default=True, type=bool)
    args = parser.parse_args()

    return args

args = parse_args() 


def main():
    logs_path = args.logs_path
    video_path = args.video_path
    restore = args.restore
    train = args.train

    # Initial PLE environment
    os.putenv('SDL_VIDEODRIVER', 'fbcon')
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    # Design reward
    reward_values = {
        "positive": 1, 
        "tick": 0.1, 
        "loss": -1,  
    }
    env = PLE(FlappyBird(), fps=30, display_screen=False, reward_values=reward_values)
    action_set = env.getActionSet()
    
    reply_buffer = Reply_Buffer(Config.reply_buffer_size)
    agent = Agent(action_set)

    reward_logs = []
    loss_logs = []

    # restore model
    if restore:
        agent.restore(restore)

    for episode in range(1, Config.total_episode+1):
        # reset env
        env.reset_game()
        env.act(0)
        obs = convert(env.getScreenGrayscale())
        state = np.stack([[obs for _ in range(4)]], axis=0)
        t_alive = 0
        total_reward = 0
        
        if episode % Config.save_video_frequency == 0 and episode > Config.initial_observe_episode: 
            agent.stop_epsilon()
            frames = [env.getScreenRGB()] 
    
        while not env.game_over():
            action = agent.take_action(state)
            reward = env.act(action_set[action])
            if episode % Config.save_video_frequency == 0 and episode > Config.initial_observe_episode: 
                frames.append(env.getScreenRGB()) 
            obs = convert(env.getScreenGrayscale())
            obs = np.reshape(obs, [1, 1, obs.shape[0], obs.shape[1]])
            state_new = np.append(state[:, 1:,...], obs, axis=1)
            action_onehot = np.zeros(len(action_set))
            action_onehot[action] = 1
            t_alive += 1
            total_reward += reward
            reply_buffer.append((state, action_onehot, reward, state_new, env.game_over()))
            state = state_new
        
        # save video
        if episode % Config.save_video_frequency == 0 and episode > Config.initial_observe_episode: 
            os.makedirs(video_path, exist_ok=True)
            clip = make_video(frames, fps=60).rotate(-90)
            clip.write_videofile(os.path.join(video_path, 'env_{}.mp4'.format(episode)), fps=60)
            agent.restore_epsilon()
            print('Episode: {} t: {} Reward: {:.3f}' .format(episode, t_alive, total_reward))
  
        if episode > Config.initial_observe_episode and train:
            # save model
            if episode % Config.save_logs_frequency == 0:
                agent.save(episode, logs_path)
                np.save(os.path.join(logs_path, 'loss.npy'), np.array(loss_logs))
                np.save(os.path.join(logs_path, 'reward.npy'), np.array(reward_logs))
        
            # update target network
            if episode % Config.update_target_frequency == 0:
                agent.update_target_network()
            
            # sample batch from reply buffer 
            batch_state, batch_action, batch_reward, batch_state_new, batch_over = reply_buffer.sample(Config.batch_size)
            
            # update policy network
            loss = agent.update_Q_network(batch_state, batch_action, batch_reward, batch_state_new, batch_over)
        
            loss_logs.extend([[episode, loss]]) 
            reward_logs.extend([[episode, total_reward]]) 
        
            # print reward and loss
            if episode % Config.show_loss_frequency == 0: 
                print('Episode: {} t: {} Reward: {:.3f} Loss: {:.3f}' .format(episode, t_alive, total_reward, loss))
        
            agent.update_epsilon()


if __name__ == "__main__":
    print(args)
    main()
     
     
