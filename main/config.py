import os
import cv2

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Paths
VIDEO_URLS = [
    'http://192.168.0.111:81/stream',  # Camera 1 (no fisheye) front
    'http://192.168.0.112:81/stream',  # Camera 2 (no fisheye) back
    'http://192.168.0.114:81/stream',  # Camera 3 ( fisheye)    		
    'http://192.168.0.113:81/stream'   # Camera 4 ( fisheye) 
    #'http://192.168.0.115:81/stream'
    #r"./vids/recording(f).mp4",
    #r"./vids/recording(b).mp4",
    #r"./vids/recording(l).mp4",
    #r"./vids/recording(r).mp4"
    #r"./recording8.mp4"
    #r"./recording8.mp4",
    #r"./recording8.mp4",
    #r"./recording8.mp4"
    #r"./recordings/save8/camera2/recording5.avi",
    #r"./recordings/save8/camera2/recording6.avi",
    #r"./recordings/save8/camera2/recording7.avi",
    #r"./vids/recording5.avi",
    #r"./vids/recording6.avi",
    #r"./vids/recording7.avi",
    #r"./vids/recording8.avi"


    #"http://192.168.0.100:81/stream",
    #"http://192.168.0.102:81/stream",
    #"http://192.168.0.104:81/stream"

 ]    
# Frame and speed parameters"""  """
FPS = 30
FRAME_TIME = 1 / FPS
#REAL_WORLD_WIDTH = 1  # (in meters) need to calibrate
#PIXELS_PER_METER = 296 / REAL_WORLD_WIDTH # need to calibrate
MIN_DISTANCE = 0.1  # Minimum pixel distance to avoid noise

