# Alyssa Hammonds

# Assignment 1: Color Spaces and Data Augmentation
import sys
import numpy as np
import cv2 as cv

def rgb2hsv(img, r, g, b):
    # Normalize RGB values to be in range of [0,1]    

    # V = max(R, G, B)
    V = np.max(img, axis=2)

    # C = V - min(R, G, B)
    C = V - np.min(img, axis=2)

    # S = C / V
    if (V == 0).any():
        S = np.zeros_like(V)
    else:
        S = C / V

    # getting run-time errors
    H = np.zeros_like(V)
    non_zero_C = C != 0
    H = np.where(non_zero_C & (V == r), 60 * (g - b) / C, H)
    H = np.where(non_zero_C & (V == g), 60 * (b - r) / C + 120, H)
    H = np.where(non_zero_C & (V == b), 60 * (r - g) / C + 240, H)
  

    return H, S, V

def hsv2rbg(H, S, V):
    H = np.where(h >= 360, h-360, h)
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

    #normalize rgb values
    img = img_array / 255
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]

    # convert the image to HSV
    H, S, V = rgb2hsv(img, r, g, b)

    # modify the HSV values
    H += h
    S += s
    V += v

    # convert the image back to RGB
    R, G, B = hsv2rbg(H, S, V)

    # save the image as jpg
    img = np.dstack((R, G, B))
    img = img * 255
    img = img.astype(np.uint8)
    cv.imwrite("modified_image.jpg", img)
    
    print("Image saved as", "modified_image.jpg")
    


