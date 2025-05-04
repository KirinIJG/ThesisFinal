import os
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

VIDEO_URLS = [
    'http://192.168.0.111:81/stream',  # Camera 1 (no fisheye) front
    'http://192.168.0.112:81/stream',  # Camera 2 (no fisheye) back
    'http://192.168.0.114:81/stream',  # Camera 3 (fisheye) right		
    'http://192.168.0.113:81/stream'   # Camera 4 (fisheye) left
 ]    

FPS = 30
FRAME_TIME = 1 / FPS
MIN_DISTANCE = 0.1 

