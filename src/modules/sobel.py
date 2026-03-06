from PIL import Image
from modules.histogram import histogram_exp
import numpy as np

def hor_sobel(r, g, b, mask : np.ndarray, *, a = None): #TODO Fazer a analise quando tiver o alfa
    im_sobel_ver = correlation()
    histogram_exp(im_sobel_ver)
    ...

def ver_sobel(r, g, b, mask : np.ndarray, *, a = None): #TODO Fazer a analise quando tiver o alfa
    im_sobel_ver = correlation()
    histogram_exp()
    ...

def sobel(r, g, b, mask : list[list[int]], *, a = None): #TODO Fazer a analise quando tiver o alfa
    mask_ver = np.array(mask)
    mask_hor = np.transpose(mask)

    im_sobel_hor = hor_sobel(r, g, b, mask_hor)
    im_sobel_ver = ver_sobel(r, g, b, mask_ver)
    
    im_sobel_hor_arr = np.asarray(im_sobel_hor)
    im_sobel_ver_arr = np.asarray(im_sobel_ver)

    im_sobel_arr = im_sobel_hor_arr + im_sobel_ver_arr

    im_sobel = Image.fromarray(im_sobel_arr)

    return im_sobel