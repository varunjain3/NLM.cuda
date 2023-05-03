import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import cv2
import time


def show_im(img):
    plt.imshow(img, cmap="gray")
    plt.show()


def MSE(Y, YH):
    Y = Y.astype(float)
    YH = YH.astype(float)
    return np.square(Y - YH).mean()


def PSNR(original, noisy, peak=100):
    mse = np.mean((original-noisy)**2)
    return 10*np.log10(peak*peak/mse)


# To get a window where center is (x,y) that is of size (N,N)


def get_window(img, x, y, N=25):
    """
    Extracts a small window of input image, around the center (x,y)
    img - input image
    x,y - cordinates of center
    N - size of window (N,N) {should be odd}
    """

    h, w, c = img.shape             # Extracting Image Dimensions

    arm = N//2                      # Arm from center to get window
    window = np.zeros((N, N, c))
    # print((0, x-arm))
    xmin = max(0, x-arm)
    xmax = min(w, x+arm+1)
    ymin = max(0, y-arm)
    ymax = min(h, y+arm+1)

    window[arm - (y-ymin):arm + (ymax-y), arm - (x-xmin)
                  :arm + (xmax-x)] = img[ymin:ymax, xmin:xmax]

    # if np.max(window)>0:
    #     print()
    return window

# The main function
def NL_means(img,h=8.5,f=6,t=11):
    # neighbourhood size 2f+1
    N = 2*f + 1

    # sliding window size 2t+1
    S = 2*t + 1

    # Filtering Parameter
    sigma_h = h

    # Padding the image
    pad_img = np.pad(img, t+f)

    # Getting the height and width of the image
    h, w = img.shape
    h_pad, w_pad = pad_img.shape

    neigh_mat = np.zeros((h+S-1, w+S-1, N, N))

    # Making a dp neighbourhood for all pixels (used for vectorizing sliding window algorithm)
    for y in range(h+S-1):
        for x in range(w+S-1):
            neigh_mat[y, x] = np.squeeze(get_window(
                pad_img[:, :, np.newaxis], x+f, y+f, 2*f+1))

    # Empty image to be filled by the algorithm
    output = np.zeros(img.shape)

    # Initializing the counter
    prog = tqdm(total=(h-1)*(w-1), position=0, leave=True)

    # Iterating for each pixel
    for Y in range(h):
        for X in range(w):
            # Shifting for padding
            x = X + t
            y = Y + t
            # Getting neibourhood in chunks of search window
            a = get_window(np.reshape(
                neigh_mat, (h+S-1, w+S-1, N*N)), x, y, S)
            
            # Getting self Neigbourhood
            b = neigh_mat[y, x].flatten()

            # Getting distance of vectorized neibourhood
            c = a-b

            # Determining weights
            d = c*c
            e = np.sqrt(np.sum(d, axis=2))
            F = np.exp(-e/(sigma_h*sigma_h))

            # Summing weights
            Z = np.sum(F)

            # Calculating average pixel value
            im_part = np.squeeze(get_window(pad_img[:, :, None], x+f, y+f, S))
            NL = np.sum(F*im_part)
            output[Y, X] = NL/Z
            
            # Updating counter
            prog.update(1)
    return output

import sys

def main():
    
    # If called using seprate entity.
    name = "Image3.png"

    img = np.array(ImageOps.grayscale(Image.open(name)), dtype=float)    
    img = cv2.resize(img, (int(sys.argv[1]), int(sys.argv[1])))
    
    start = time.time()
    nlmeans_img = NL_means(img)
    end = time.time()
    print(f"Time taken: {int((end-start)*1000)}ms")
    
    cv2.imwrite("OutputImage.png",nlmeans_img)
        
if __name__ == "__main__":
    main()