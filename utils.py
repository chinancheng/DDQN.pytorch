import numpy as np
import cv2
from config import Config 


def make_video(images, fps):

    import moviepy.editor as mpy
    duration = len(images) / fps

    def make_frame(t):
        x = images[int(len(images) / duration * t)]
        
        return x.astype(np.uint8)
    
    clip = mpy.VideoClip(make_frame, duration=duration)
    clip.fps = fps
        
    return clip

def convert(image):
    image = cv2.resize(image, (Config.screen_height, Config.screen_width))
    _, image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY_INV)
    
    return image
