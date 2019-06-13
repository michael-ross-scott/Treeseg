import cv2
import numpy as np


# TODO need to get format into h5 files
def hough_transform(image):
    # Read image, cwd is in Treeseg
    img = cv2.imread("Dave", cv2.IMREAD_COLOR)  # road.png is the filename
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


# TODO template matching with hd5 format
def template_transform(image):
    img_rgb = cv2.imread('Dave.png')
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('template.png', 0)
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    cv2.imwrite('Image.png', img_rgb)

