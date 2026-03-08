from PIL import Image
from modules.histogram import histogram_exp
from modules.correlation import correlation 
import numpy as np

def hor_sobel(r, g, b, im_size : tuple[int, int], mask : list[list[int]], mask_size : tuple[int, int], stride : int, activation_function : str, *, a = None): #TODO Fazer a analise quando tiver o alfa
    im_sobel_hor = correlation(r, g, b, im_size, mask, mask_size, stride, activation_function, a=a)
    
    return im_sobel_hor    

def ver_sobel(r, g, b, im_size : tuple[int, int], mask : list[list[int]], mask_size : tuple[int, int], stride : int, activation_function : str, *, a = None): #TODO Fazer a analise quando tiver o alfa
    im_sobel_ver = correlation(r, g, b, im_size, mask, mask_size, stride, activation_function, a=a)

    return im_sobel_ver


def sobel(r, g, b, im_size : tuple[int, int], mask : list[list[int]], mask_size : tuple[int, int], stride : int, activation_function : str, *, a = None) -> Image.Image:
    mask_hor = np.array(mask)
    mask_ver = np.transpose(mask)

    im_sobel_hor = hor_sobel(r, g, b, im_size, mask_hor, mask_size, stride, activation_function, a=a)
    im_sobel_ver = ver_sobel(r, g, b, im_size, mask_ver, mask_size, stride, activation_function, a=a)

    im_sobel_hor_arr = np.asarray(im_sobel_hor)
    im_sobel_ver_arr = np.asarray(im_sobel_ver)

    im_sobel_arr = np.clip(im_sobel_hor_arr + im_sobel_ver_arr, 0, 255).astype(np.uint8)
    im_sobel = Image.fromarray(im_sobel_arr)

    im_sobel = histogram_exp(im_sobel)

    return im_sobel