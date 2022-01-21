import numpy as np



def S_function(X,lmbd):
    """S function defined to contrast the details in the high frequency components of the L channel of the image"""
    return (1 / (1+np.exp(-lmbd*X)) - 1 / 2) / (1 / (1+np.exp(-lmbd*255)) - 1 /2 ) * 255


def process_depth_v0(L_low):
    """Applies the histogram equalization to the background and foreground as described in the 
    paper for the low frequency components of the L channel of the image. IN PLACE"""
    def sum_var(img,T_opt):
        h,w = img.shape
        w1 = np.sum(img <= T_opt)/h/w
        w2 = np.sum(img > T_opt)/h/w

        if w1 == 0:
            return w2 * np.mean(img[img>T_opt])
        elif w2 == 0:
            return w1 * np.mean(img[img<=T_opt])

        u1 = np.mean(img[img<=T_opt])
        u2 = np.mean(img[img>T_opt])
        u = np.mean(img)

        return w1 * (u1 - u)**2 + w2 * (u2 - u)**2

    T_v = []
    for T in range(np.min(L_low),np.max(L_low)):
        T_v.append(sum_var(L_low, T))

    T_opt = np.nanargmax(T_v)

    I0 = np.min(L_low[L_low<=T_opt])
    I1 = np.max(L_low[L_low<=T_opt])
    I2 = np.max(L_low[L_low>T_opt])

    n_D = np.sum(L_low<=T_opt)
    n_F = np.sum(L_low>T_opt)


    I_n = [np.sum([L_low[L_low<=T_opt] <= i]) for i in range(T_opt+1)]
    I_n_f = [np.sum([L_low[L_low>T_opt] <= i]) for i in range(T_opt+1,255)]
    
    def equalize_background(pixel):
        return I0 + (I1 - I0) * I_n[pixel] / n_D

    
    
    def equalize_foreground(pixel):
        return I1 + 1 + (I2 - 1 - I1) * I_n_f[pixel-T_opt-1] / n_F

    for i in range(L_low.shape[0]):
        for j in range(L_low.shape[1]):
            if L_low[i,j]<=T_opt:
                L_low[i,j] = equalize_background(L_low[i,j])
            else:
                L_low[i,j] = equalize_foreground(L_low[i,j])
