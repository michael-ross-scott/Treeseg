import cv2
import numpy as np
import math


# Read image, cwd is in Treeseg
img = cv2.imread("Dave.png", cv2.IMREAD_COLOR)  # road.png is the filename
# Convert the image to gray-scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Find the edges in the image using canny detector
edges = cv2.Canny(gray, 50, 200)
# Detect points that form a line
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 20)
# Draw lines on the image
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
# write result
cv2.imwrite("Image.png", img)

#def hough_transform(image):


def template_transform(image):
    pass
