#CANNY EDGE DETECTOR IMPORTS
import cv2
import numpy as np
from matplotlib import pyplot as plt

#MEAN SHIFT IMPORTS


def mean_shift_transform(image):
    pass

########### MEAN SHIFT METHODS ###########

def edge_detector_transform(image):
    image = cv2.imread('C:/Users/charl_6lfr58n/Desktop/2.jpg', 0)
    edges = cv2.Canny(image, 100, 200)

    plt.subplot(121), plt.imshow(image, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    plt.show()
