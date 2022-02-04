from skimage.restoration import denoise_bilateral
from skimage.morphology import area_closing
import numpy as np
import skimage.filters as fil


def S_function(X, lmbd):
    """S function defined to contrast the details in the high frequency components of the L channel of the image"""
    return (1 / (1 + np.exp(-lmbd * X)) - 1 / 2) / (1 / (1 + np.exp(-lmbd * 255)) - 1 / 2) * 255


def sum_var(img, T_opt):
    h, w = img.shape
    w1 = np.sum(img <= T_opt) / h / w
    w2 = np.sum(img > T_opt) / h / w

    if w1 == 0:
        return w2 * np.mean(img[img > T_opt])
    elif w2 == 0:
        return w1 * np.mean(img[img <= T_opt])

    u1 = np.mean(img[img <= T_opt])
    u2 = np.mean(img[img > T_opt])
    u = np.mean(img)

    return w1 * (u1 - u) ** 2 + w2 * (u2 - u) ** 2


def process_depth_v0(L_low):
    """Applies the histogram equalization to the background and foreground as described in the 
    paper for the low frequency components of the L channel of the image. IN PLACE"""

    T_v = []
    for T in range(np.min(L_low), np.max(L_low)):
        T_v.append(sum_var(L_low, T))

    T_opt = np.nanargmax(T_v)

    I0 = np.min(L_low[L_low <= T_opt])
    I1 = np.max(L_low[L_low <= T_opt])
    I2 = np.max(L_low[L_low > T_opt])

    n_D = np.sum(L_low <= T_opt)
    n_F = np.sum(L_low > T_opt)

    I_n = [np.sum([L_low[L_low <= T_opt] <= i]) for i in range(T_opt + 1)]
    I_n_f = [np.sum([L_low[L_low > T_opt] <= i]) for i in range(T_opt + 1, 255)]

    def equalize_background(pixel):
        return I0 + (I1 - I0) * I_n[pixel] / n_D

    def equalize_foreground(pixel):
        return I1 + 1 + (I2 - 1 - I1) * I_n_f[pixel - T_opt - 1] / n_F

    for i in range(L_low.shape[0]):
        for j in range(L_low.shape[1]):
            if L_low[i, j] <= T_opt:
                L_low[i, j] = equalize_background(L_low[i, j])
            else:
                L_low[i, j] = equalize_foreground(L_low[i, j])


# Enhancement of the background detection
def compute_B(I, n=4):
    diff = np.array(
        [
            np.abs(I - fil.gaussian(I, ((2 ** i * n + 1) ** 2 - 1) / 12)) for i in range(n)
        ]
    )
    B = np.mean(
        diff, axis=0
    )
    return B.reshape((I.shape[0], I.shape[1]))


def compute_t_map(B, window=7):
    h, w = B.shape
    t = np.zeros_like(B)
    neigh = (window - 1) // 2
    B = np.pad(B, neigh)

    for i in range(neigh, h + neigh):
        for j in range(neigh, w + neigh):
            t[i - neigh, j - neigh] = np.max(B[i - neigh:i + neigh, j - neigh:j + neigh])
    return t


def compute_final_map(t):
    t_f = area_closing(t, 256, connectivity=2)
    t_ff = denoise_bilateral(t_f, sigma_spatial=2, sigma_color=2)
    return t_ff


def process_depth_v1(L_low, n=4, window=7):
    B = compute_B(L_low, n)
    t = compute_t_map(B, window)
    t = compute_final_map(t)

    T_v = []
    for T in np.linspace(np.min(t), np.max(t), 100):
        T_v.append(sum_var(t, T))

    T_opt = np.linspace(np.min(t), np.max(t), 100)[np.argmax(T_v)]

    L_low_b = L_low * (t <= T_opt)
    L_low_f = L_low * (t > T_opt)

    I0b = np.min(L_low_b)
    I1b = np.max(L_low_b)
    I0f = np.max(L_low_f)
    I1f = np.max(L_low_f)

    n_D = np.sum(t <= T_opt)
    n_F = np.sum(t > T_opt)

    I_n = [np.sum(L_low_b <= i) for i in range(255)]
    I_n_f = [np.sum(L_low_f <= i) for i in range(255)]

    def equalize_background(pixel):
        return I0b + (I1b - I0b) * I_n[pixel] / n_D

    def equalize_foreground(pixel):
        return I0f + (I1f - I0f) * I_n_f[pixel] / n_F

    for i in range(L_low.shape[0]):
        for j in range(L_low.shape[1]):
            if t[i, j] <= T_opt:
                L_low[i, j] = equalize_background(L_low[i, j])
            else:
                L_low[i, j] = equalize_foreground(L_low[i, j])


def fft_from_img(img):
    return np.fft.fftshift(np.fft.fft2(img))


def reconstruct_from_fft(img_fft):
    img = abs(np.fft.ifft2(img_fft))
    return img.clip(0, 255)


def draw_circle(shape, diameter):
    """
    Input:
    shape    : tuple (height, width)
    diameter : scalar

    Output:
    np.array of shape  that says True within a circle with diamiter =  around center
    """
    assert len(shape) == 2
    filter = np.zeros(shape, dtype=np.bool)
    center = np.array(filter.shape) / 2.0

    for iy in range(shape[0]):
        for ix in range(shape[1]):
            filter[iy, ix] = (iy - center[0]) ** 2 + (ix - center[1]) ** 2 < diameter ** 2
    return filter


def get_filtered_channel(filter, fft_img_channel):
    filtered = np.zeros(fft_img_channel.shape[:2], dtype=complex)
    filtered[filter] = fft_img_channel[filter]
    return filtered


def get_filtered_img(filter, img):

    fft_img = fft_from_img(img)
    filtered = get_filtered_channel(filter, fft_img)
    img_reco = reconstruct_from_fft(filtered)
    return img_reco.astype(int)


def get_filtered_img_test(filter, img):
    fft_img_channel_0 = fft_from_img(img[:, :, 0])
    fft_img_channel_1 = fft_from_img(img[:, :, 1])
    fft_img_channel_2 = fft_from_img(img[:, :, 2])

    filtered_red_channel = get_filtered_channel(filter, fft_img_channel_0)
    filtered_green_channel = get_filtered_channel(filter, fft_img_channel_1)
    filtered_blue_channel = get_filtered_channel(filter, fft_img_channel_2)

    red_channel = reconstruct_from_fft(filtered_red_channel)
    green_channel = reconstruct_from_fft(filtered_green_channel)
    blue_channel = reconstruct_from_fft(filtered_blue_channel)

    filtered_img = np.dstack([red_channel.astype(int),
                              green_channel.astype(int),
                              blue_channel.astype(int)])
    return filtered_img
