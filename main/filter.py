import numpy as np

from skimage import io
import matplotlib.pyplot as plt
import skimage.color as col 
import skimage.filters as fil 
from main.utils import S_function, process_depth_v0, process_depth_v1, process_depth_v2

PATH = 'raw-890/'


def first_color_correction(img, a1=0.001, a2=0.995):
    """Applies the first transformation to the RGB channels of the image. IN PLACE"""
    sums = np.sum(img, axis=(0,1))

    P = np.max(sums)/sums

    rc = np.concatenate([
        (a1*P).reshape(-1,1),
        (a2*P).reshape(-1,1)
        ], axis=1)
    t1 = [np.quantile(img[:,:,i], min(1,rc[i,0])) for i in range(3)]
    t2 = [np.quantile(img[:,:,i], min(1,rc[i,1])) for i in range(3)]

    for i in range(3):
        img[:,:,i] = np.clip(img[:,:,i], t1[i], t2[i])
        img[:,:,i] =  ((img[:,:,i] - t1[i])/(t2[i]-t1[i]) * 255).astype('int32')



def second_correction(img,sigma=0.3, lmbd = 0.01):
    """Applies the second transformation by looking at the image in the LAB space. IN PLACE"""
    img = col.rgb2lab(img)
    L = img[:,:,0]

    L_low = fil.gaussian(L, sigma).astype('int32')
    L_high = L - L_low
    
    L_high = S_function(L_high, lmbd).astype('int32')
    process_depth_v0(L_low)

    img[:,:,0] = np.clip(((L_low+L_high) * (L_low>=0) * (255 * ((L_low>255) + (L_high > 255)) + 1)),0,255)
    img = col.lab2rgb(img)
    return img



def full_filter(img, sigma=0.3, lmbd=0.01, a1=1e-3, a2=0.995):
    """Filters an image."""
    filt_img = img.copy()
    first_color_correction(filt_img, a1, a2)
    return second_correction(filt_img, sigma, lmbd)


def second_correction_v1(img,sigma=0.7, lmbd = 0.01, n=4, window=7, sp=10, sc=2):
    """Applies the second transformation by looking at the image in the LAB space. """
    img = col.rgb2lab(img)
    L = img[:,:,0]

    L_low = fil.gaussian(L, sigma).astype('int32')
    L_high = L - L_low
    
    L_high = S_function(L_high, lmbd).astype('int32')
    
    process_depth_v1(L_low,n=n, window=window, sp=sp, sc=sc)

    img[:,:,0] = np.clip(((L_low+L_high) * (L_low>=0) * (255 * ((L_low>255) + (L_high > 255)) + 1)),0,255)
    img = col.lab2rgb(img)
    return img


def full_filterv1(img, sigma=0.3, lmbd=0.01, a1=1e-3, a2=0.995, n=4, window=7, sp=10, sc=2):
    """Filters an image."""
    filt_img = img.copy()
    first_color_correction(filt_img, a1, a2)
    return second_correction_v1(filt_img, sigma, lmbd, n , window, sp, sc)





def second_correction_v2(img,sigma=0.7, lmbd = 0.0001, n=4, window=7, sp=10, sc=2):
    """Applies the second transformation by looking at the image in the LAB space. """
    img = col.rgb2lab(img)
    L = img[:,:,0]

    L_low = fil.gaussian(L, sigma).astype('int32')
    L_high = L - L_low
    
    L_high = S_function(L_high, lmbd).astype('int32')
    
    process_depth_v2(L_low,n=n, window=window, sp=sp, sc=sc)

    img[:,:,0] = np.clip(((L_low+L_high) * (L_low>=0) * (255 * ((L_low>255) + (L_high > 255)) + 1)),0,255)
    img = col.lab2rgb(img)
    return img


def full_filterv2(img, sigma=0.3, lmbd=0.01, a1=1e-3, a2=0.995, n=4, window=7, sp=10, sc=2):
    """Filters an image."""
    filt_img = img.copy()
    first_color_correction(filt_img, a1, a2)
    return second_correction_v2(filt_img, sigma, lmbd, n , window, sp, sc)