import cv2
import os

def video_into_frames( video_path,saved_path ):

    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    while success:
      cv2.imwrite(os.path.join(saved_path , "frame%d.jpg" % count), image) # save frame as JPEG file
      success,image = vidcap.read()
      print ('Read a new frame: ', success)
      count += 1
      
