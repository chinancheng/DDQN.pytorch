# Double Deep Q Learning (DDQN) In PyTorch
DDQN inplementation on PLE FlappyBird environment in PyTorch.  
  
<img src='./assets/DDQN.gif'>  
  
**DDQN** is proposed to solve the overestimation issue of Deep Q Learning (DQN). Apply separate target network to choose action, reducing the correlation of action selection and value evaluation.

## Requirement
* Python 3.6
* Pytorch
* [PLE (PyGame-Learning-Environment)](https://github.com/ntasfi/PyGame-Learning-Environment) 
* Moviepy

 ## Usage
 * Train 
 ```
 python main.py --train=True --video_path=./video --logs_path=./logs 
 ```
 * Restore Pretrain Model
 ```
 python main.py --restore=./pretrain/model-98500.pth  
 ```
 * HyperParameter in `config.py`
 
 ## Result 
 * [Full Video (with 60 FPS)](https://www.youtube.com/watch?v=GCHTadB22P8&feature=youtu.be)
  
 ## Reference
 * [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
 * [CS 294: Deep Reinforcement Learning](http://rll.berkeley.edu/deeprlcourse/)
  * [DeepLearningFlappyBird](https://github.com/yenchenlin/DeepLearningFlappyBird)
