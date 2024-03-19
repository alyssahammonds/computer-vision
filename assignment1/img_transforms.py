# Alyssa Hammonds

# Assignment 1: Color Spaces and Data Augmentation
import sys
import numpy as np
import cv2 as cv
import random as rand

# from color_space_test.py
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

def random_crop(img, size):
    # check if the size is valid
    if size <= 0 or size > min(img.shape[:2]):
        raise ValueError("Invalid crop size")

    # generate random coordinates
    center_x = np.random.randint(size // 2, img.shape[1] - size // 2)
    center_y = np.random.randint(size // 2, img.shape[0] - size // 2)

    # calculate (x,y) 
    x1 = center_x - size // 2
    x2 = x1 + size
    y1 = center_y - size // 2
    y2 = y1 + size

    rand_crop = img[y1:y2, x1:x2]

    return rand_crop

def extract_patch(img, num_patches):
    # check if the number of patches is valid
    if num_patches <= 0:
        raise ValueError("Invalid number of patches")
    
    img_size = img.shape[0]
    patch_size = img_size // num_patches


    patches = []
    for i in range(num_patches):
        for j in range(num_patches):
            x1 = i * patch_size
            x2 = x1 + patch_size
            y1 = j * patch_size
            y2 = y1 + patch_size
            patch = img[y1:y2, x1:x2]
            patches.append(patch)

    return patches

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

def color_jitter(img, hue, saturation, value):
    # random hue
    hue = rand.randint(0, hue) # shouldnt be greater than given input
    # random saturation
    saturation = rand.uniform(0, saturation)
    # random value
    value = rand.randint(0, value) 

    # convert to HSV
    img = img / 255
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    H, S, V = rgb2hsv(img, r, g, b)

    # modify the HSV values
    H += hue
    S += saturation
    V += value

    # convert the image back to RGB
    R, G, B = hsv2rbg(H, S, V)

    img = np.dstack((R, G, B))
    img = img * 255
    img = img.astype(np.uint8)
    return img

if __name__ == "__main__":

    # ask for the image file and the size of the crop
    filename = input("Enter the filename of the image: ")
    size = int(input("Enter the size of the crop: "))

    img = cv.imread(filename)
    img_array = np.array(img)
    rand_crop = random_crop(img, size)

    # save the cropped image
    cv.imwrite("cropped_image.jpg", rand_crop)
    print("Image saved as cropped_image.jpg")

    # ask for the number of patches
    num_patches = int(input("Enter the number of patches: "))
    patches = extract_patch(img, num_patches)

    # from color_space_test.py
    h = int(input("Enter the hue modifier: "))
    if h < 0 or h > 360:
        print("Invalid input")
        exit()
    s = float(input("Enter the saturation modifier: "))
    if s < 0 or s > 1:
        print("Invalid input")
        exit()
    v = int(input("Enter the value modifier: "))

    color_jitter = color_jitter(img, h, s, v)
    cv.imwrite("color_jitter.jpg", color_jitter)
    print("Image saved as color_jitter.jpg")
 