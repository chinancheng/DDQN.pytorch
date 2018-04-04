import argparse
import os
import argparse
import numpy as np
from visdom import Visdom

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--logs_path', dest='logs_path', help='path of the checkpoint folder',
                        default='./logs', type=str)
    args = parser.parse_args()
    
    return args

args = parse_args()

def main():
    logs_path = args.logs_path
    episode, reward = zip(*np.load(os.path.join(logs_path, 'reward.npy')))
    _, loss = zip(*np.load(os.path.join(logs_path, 'loss.npy')))

    avg_reward = np.cumsum(reward) / np.arange(1, len(reward) + 1)

    viz = Visdom(env='main', port=8787)
    viz.line(
        X=np.array(step).reshape(-1, 1).repeat(3, 1),
        Y=np.column_stack([reward, avg_reward, loss]),
        opts=dict(
            legend=['Reward', 'Average Reward', 'Loss']
        )
    )


if __name__ == '__main__':
    main()
