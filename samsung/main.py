import re
import cv2
import numpy as np

game_img = cv2.imread('samsung_game.png', cv2.IMREAD_UNCHANGED)
cross_img = cv2.imread('cross.png', cv2.IMREAD_UNCHANGED)

# cv2.imshow('Game', game_img)
# cv2.waitKey()
# cv2.destroyAllWindows()

# cv2.imshow('Needle', wheat_img)
# cv2.waitKey()
# cv2.destroyAllWindows()

# There are 6 comparison methods to choose from:
# TM_CCOEFF, TM_CCOEFF_NORMED, TM_CCORR, TM_CCORR_NORMED, TM_SQDIFF, TM_SQDIFF_NORMED
# You can see the differences at a glance here:
# https://docs.opencv.org/master/d4/dc6/tutorial_py_template_matching.html
result = cv2.matchTemplate(game_img, cross_img, cv2.TM_CCOEFF_NORMED)

# cv2.imshow('Result', result)
# cv2.waitKey()
# cv2.destroyAllWindows()

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

print(max_val)

w = cross_img.shape[1]
h = cross_img.shape[0]

# cv2.rectangle(game_img, max_loc,
#               (max_loc[0] + w, max_loc[1] + h), (0, 255, 255), 2)

# cv2.imshow('Game', game_img)
# cv2.waitKey()
# cv2.destroyAllWindows()

threshold = .10

yloc, xloc = np.where(result >= threshold)

len(xloc)


# What is a rectangle?
# x, y, w, h
rectangles = []
for (x, y) in zip(xloc, yloc):
    rectangles.append([int(x), int(y), int(w), int(h)])
    rectangles.append([int(x), int(y), int(w), int(h)])

rectangles, weights = cv2.groupRectangles(rectangles, 1, 0.2)

for (x, y, w, h) in rectangles:
    cv2.rectangle(game_img, (x, y), (x + w, y + h), (0, 255, 255), 2)

cv2.imshow('Game', game_img)
cv2.waitKey()
cv2.destroyAllWindows()
