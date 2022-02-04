import numpy as np
import skimage.filters as fil 

import skimage.color as col 
from main.utils import *

def visualize_depth_v0(img, sigma = 0.3):
    """Applies the histogram equalization to the background and foreground as described in the 
    paper for the low frequency components of the L channel of the image. IN PLACE"""
    img = col.rgb2lab(img)
    L = img[:, :, 0]

    L_low = fil.gaussian(L, sigma).astype('int32')

    def sum_var(img, T_opt):
        h,w = img.shape
        w1 = np.sum(img <= T_opt)/h/w
        w2 = np.sum(img > T_opt)/h/w

        if w1 == 0:
            return w2 * np.mean(img[img > T_opt])
        elif w2 == 0:
            return w1 * np.mean(img[img <= T_opt])

        u1 = np.mean(img[img <= T_opt])
        u2 = np.mean(img[img > T_opt])
        u = np.mean(img)

        return w1 * (u1 - u)**2 + w2 * (u2 - u)**2

    T_v = []
    for T in range(np.min(L_low), np.max(L_low)):
        T_v.append(sum_var(L_low, T))

    T_opt = np.nanargmax(T_v)

    return L_low>T_opt


def visualize_depth_v1(img,window =7, n=4, sigma=0.3):
    img = col.rgb2lab(img)
    L = img[:,:,0]

    L_low = fil.gaussian(L, sigma).astype('int32')

    B = compute_B(L_low,n)
    t = compute_t_map(B, window)
    t = compute_final_map(t)

    return t
