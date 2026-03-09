import math

import numpy as np
from PIL import Image


def correlation(
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
    """faz a correlacao do filtro com a imagem"""

    im_x, im_y = im_size
    mask_x, mask_y = mask_size
    mask_center_x, mask_center_y = math.floor(mask_x / 2), math.floor(mask_y / 2)

    out_h = len(range(mask_center_x, im_x - (mask_x - mask_center_x - 1), stride))
    out_w = len(range(mask_center_y, im_y - (mask_y - mask_center_y - 1), stride))

    output = np.zeros((out_w, out_h, 4 if a else 3))

    row, col = 0, 0

    for i_img in range(out_h):
        for j_img in range(out_w):
            sum_r, sum_g, sum_b, sum_a = 0, 0, 0, 0
            for i_mask in range(mask_x):
                for j_mask in range(mask_y):
                    value_r = (
                        r.getpixel(
                            (
                                i_img * stride + i_mask,
                                j_img * stride + j_mask,
                            )
                        )
                        * mask[i_mask][j_mask]
                    )
                    sum_r += value_r

                    value_g = (
                        g.getpixel(
                            (
                                i_img * stride + i_mask,
                                j_img * stride + j_mask,
                            )
                        )
                        * mask[i_mask][j_mask]
                    )
                    sum_g += value_g

                    value_b = (
                        b.getpixel(
                            (
                                i_img * stride + i_mask,
                                j_img * stride + j_mask,
                            )
                        )
                        * mask[i_mask][j_mask]
                    )
                    sum_b += value_b

                    if a:
                        value_a = (
                            a.getpixel(
                                (
                                    i_img * stride + i_mask,
                                    j_img * stride + j_mask,
                                )
                            )
                            * mask[i_mask][j_mask]
                        )
                        sum_a += value_a
            if not a:
                output[row, col] = [
                    min(max(sum_r, 0), 255) if activation_function == "relu" else sum_r,
                    min(max(sum_g, 0), 255) if activation_function == "relu" else sum_g,
                    min(max(sum_b, 0), 255) if activation_function == "relu" else sum_b,
                ]
            else:
                output[row, col] = [
                    min(max(sum_r, 0), 255) if activation_function == "relu" else sum_r,
                    min(max(sum_g, 0), 255) if activation_function == "relu" else sum_g,
                    min(max(sum_b, 0), 255) if activation_function == "relu" else sum_b,
                    min(max(sum_a, 0), 255) if activation_function == "relu" else sum_a,
                ]
            row += 1
        col += 1
        row = 0

    return output


def to_img(arr: np.ndarray) -> Image.Image:
    im_result = Image.fromarray(
        arr.astype(np.uint8), mode="RGB" if len(arr[0][0]) <= 3 else "RGBA"
    )

    return im_result
