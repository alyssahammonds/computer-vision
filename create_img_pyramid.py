# Alyssa Hammonds

# Assignment 1: Color Spaces and Data Augmentation
import sys
import numpy as np
import cv2 as cv
import random as rand

def resize_img(img, factor):
    height, width = img.shape[0], img.shape[1]

    new_height = int(height * factor)
    new_width = int(width * factor)

    # resize using numpy array
    resized_img = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    for i in range(new_height):
        for j in range(new_width):
            x = int(i / factor)
            y = int(j / factor)
            resized_img[i, j] = img[x, y]

    return resized_img

def image_pyramid(img, num_levels):
    # check if the number of levels is valid
    for i in range(num_levels):
        img = resize_img(img, 0.5)
        cv.imwrite(f"pyramid_level_{i}.jpg", img)

    return img


if __name__ == "__main__":
    # ask user for filename
    filename = input("Enter filename: ")
    img = cv.imread(filename)
    num_levels = int(input("Enter the number of levels: "))
    img = image_pyramid(img, num_levels)
    print(f"Image pyramid created with {num_levels} levels")
    