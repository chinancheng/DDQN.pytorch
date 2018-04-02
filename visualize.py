import argparse
import os
import numpy as np
from visdom import Visdom


parser = argparse.ArgumentParser()
parser.add_argument('-l', '--logs_path', default='./logs', help='path to checkpoint folder')
args = parser.parse_args()

def main():
    logs_path = args.logs_path
    steps, loss = zip(*np.load(os.path.join(logs_path, 'loss.npy')))
    loss_length = len(loss)

    avg_loss = np.cumsum(loss) / np.arange(1, len(loss) + 1)

    window_size = max(min(loss_length // 10, 100), 1)
    batch_avg_loss = np.convolve(loss, np.ones(window_size) / window_size, mode='same')
    batch_avg_loss[:window_size - 1] = np.nan
    batch_avg_loss[-(window_size - 1):] = np.nan

    viz = Visdom(env=logs_path)
    viz.line(
        X=np.array(steps).reshape(-1, 1).repeat(3, 1),
        Y=np.column_stack([losses, avg_loss, batch_avg_loss]),
        opts=dict(
            legend=['Loss', 'Average Loss', 'Batch Average Loss (Size=%d)' % window_size]
        )
    )


if __name__ == '__main__':
    print(args)
    main()
