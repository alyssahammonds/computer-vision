# Alyssa Hammonds

# Assignment 1: Color Spaces and Data Augmentation
import numpy as np
import cv2 as cv

# Color Spaces - RGB to HSV
# Create a function that converts an RGB image represented as a numpy array into HSV format.
def rgb2hsv(img, R, G, B):
    # V = max(R, G, B)
    V = np.max(img, axis=2)

    # C = V - min(R, G, B)
    C = V - np.min(img, axis=2)

    # S = C / V
    if (V == 0).any():
        S = np.zeros_like(V)
    else:
        S = C / V

    # piecewise function to calculate H
    if (C == 0):
        H = 0
    elif (V == R):
        H = 60 * ((G - B) / C)
    elif (V == G):
        H = 60 * (2 + (B - R) / C)
    elif (V == B):
        H = 60 * (4 + (R - G) / C)

    return H, S, V

def hsv2rbg(img, H, S, V):
    C = V * S
    # We then divide up the Hue into one of 6 values:
    H = H / 60
    H = H % 6
    X = C * (1 - np.abs(H % 2 - 1))

    if 0 <= H < 1:
        R, G, B = C, X, 0
    elif 1 <= H < 2:
        R, G, B = X, C, 0
    elif 2 <= H < 3:
        R, G, B = 0, C, X
    elif 3 <= H < 4:
        R, G, B = 0, X, C
    elif 4 <= H < 5:
        R, G, B = X, 0, C
    elif 5 <= H < 6:
        R, G, B = C, 0, X

    m = V - C
    R, G, B = (R + m), (G + m), (B + m)


    return R, G, B

# testing - accepts a filename, hue modifier, saturation modifier, and value modifier

if __name__ == "__main__":
    # ask for image filename    
    filename = input("Enter the filename: ")
    img = cv.imread(filename)
    img_array = np.array(img)

    # ask for the modifiers
    h = int(input("Enter the hue modifier: "))
    if h < 0 or h > 360:
        print("Invalid input")
        exit()
    s = float(input("Enter the saturation modifier: "))
    if s < 0 or s > 1:
        print("Invalid input")
        exit()
    v = int(input("Enter the value modifier: "))

    

    # convert the image to HSV
    H, S, V = rgb2hsv(img, r, g, b)

    # modify the HSV values
    H += h
    S += s
    V += v

    # convert the image back to RGB
    R, G, B = hsv2rbg(H, S, V)

    # save the image
    np.save(filename, np.dstack((R, G, B)))
    print("Image saved as", filename)
    


