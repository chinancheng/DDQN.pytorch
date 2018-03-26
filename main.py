from ple import PLE
import os 
from ple.games.pong import Pong

os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ["SDL_VIDEODRIVER"] = "dummy"

game = Pong(width=64, height=64)
p = PLE(game, fps=30, display_screen=False, force_fps=False)
p.init()
nb_frames = 1000
reward = 0.0

for f in range(nb_frames):
    if p.game_over(): #check if the game is over
        p.reset_game()

    obs = p.getScreenRGB()
    action = myAgent.pickAction(reward, obs)
    reward = p.act(action)


    
     
