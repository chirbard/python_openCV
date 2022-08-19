import cv2 as cv
import numpy as np
import os
from time import time
from windowcapture import WindowCapture
from vision import Vision
from hsvfilter import HsvFilter
from ctypes import windll

# Make program aware of DPI scaling
user32 = windll.user32
user32.SetProcessDPIAware()

# Change the working directory to the folder this script is in.
# Doing this because I'll be putting the files from each video in their own folder on GitHub
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# WindowCapture.list_window_names()

# initialize the WindowCapture class
wincap = WindowCapture('Nexus 6P')

# load trained model
cross = cv.CascadeClassifier('cascade/cascade.xml')
# load an empty Vision class
vision_cross = Vision(None)

# cross HSV filter
hsv_filter = HsvFilter(0, 255, 0, 179, 255, 255, 143, 0, 255, 0)

loop_time = time()
while (True):

    # get an updated image of the game
    screenshot = wincap.get_screenshot()

    output_image = vision_cross.apply_hsv_filter(screenshot, hsv_filter)

    # # do object detection
    # rectangles = cross.detectMultiScale(screenshot)
    rectangles = cross.detectMultiScale(output_image)

    # # draw the detection results onto the original image
    detection_image = vision_cross.draw_rectangles(screenshot, rectangles)
    # detection_image = vision_cross.draw_rectangles(output_image, rectangles)

    # # display the images
    cv.imshow('Matches', detection_image)
    # cv.imshow('Matches', output_image)

    # debug the loop rate
    print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()

    # press 'q' with the output window focused to exit.
    # waits 1 ms every loop to process key presses
    key = cv.waitKey(1)
    if key == ord('q'):
        cv.destroyAllWindows()
        break
    elif key == ord('f'):
        cv.imwrite('positive/{}.jpg'.format(loop_time), output_image)
    elif key == ord('d'):
        cv.imwrite('negative/{}.jpg'.format(loop_time), output_image)


print('Done.')
