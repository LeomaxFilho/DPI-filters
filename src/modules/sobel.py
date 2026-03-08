import numpy as np
from PIL import Image

from modules.correlation import correlation
from modules.histogram import histogram_exp


def axis_sobel(
    r,
    g,
    b,
    im_size: tuple[int, int],
    mask: np.ndarray,
    mask_size: tuple[int, int],
    stride: int,
    activation_function: str,
    *,
    a=None,
) -> np.ndarray:
    im_sobel_ver = correlation(
        r, g, b, im_size, mask, mask_size, stride, activation_function, a=a
    )

    return im_sobel_ver


def full_axis_sobel(
    r,
    g,
    b,
    im_size: tuple[int, int],
    mask: np.ndarray,
    mask_size: tuple[int, int],
    stride: int,
    activation_function: str,
    *,
    a=None,
) -> Image.Image:
    arr_im_sobel_ver = correlation(
        r, g, b, im_size, mask, mask_size, stride, activation_function, a=a
    )
    im_sobel_ver = histogram_exp(arr_im_sobel_ver)

    return im_sobel_ver


def sobel(
    r,
    g,
    b,
    im_size: tuple[int, int],
    mask: np.ndarray,
    mask_size: tuple[int, int],
    stride: int,
    activation_function: str,
    *,
    a=None,
) -> Image.Image:
    mask_hor = np.array(mask)
    mask_ver = np.transpose(mask)

    arr_im_sobel_hor = axis_sobel(
        r, g, b, im_size, mask_hor, mask_size, stride, activation_function, a=a
    )
    arr_im_sobel_ver = axis_sobel(
        r, g, b, im_size, mask_ver, mask_size, stride, activation_function, a=a
    )

    im_sobel_arr = np.abs(arr_im_sobel_hor) + np.abs(arr_im_sobel_ver)

    im_sobel = histogram_exp(im_sobel_arr)

    im_sobel.show()

    return im_sobel
