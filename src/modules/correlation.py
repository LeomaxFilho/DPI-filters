import math
import numpy as np
from PIL import Image

def correlation(r, g, b, im_size : tuple[int, int], mask : list[list[int]], mask_size : tuple[int, int], stride : int, activation_function : str):
    im_x, im_y = im_size
    mask_x, mask_y = mask_size
    mask_center_x, mask_center_y = math.floor(mask_x/2), math.floor(mask_y/2)

    out_h = len(range(mask_center_x, im_x - (mask_x - mask_center_x - 1), stride))
    out_w = len(range(mask_center_y, im_y - (mask_y - mask_center_y - 1), stride))

    output = np.zeros((out_h, out_w, 3))
    row, col = 0, 0
    
    for i_img in range(mask_center_x, im_x - (mask_x - mask_center_x - 1), stride):
        for j_img in range(mask_center_y, im_y - (mask_y - mask_center_y - 1), stride):
            sum_r, sum_g, sum_b = 0, 0, 0
            for i_mask in range(mask_x):
                for j_mask in range(mask_y):
                    value_r = r.getpixel((i_img - mask_center_x + i_mask, j_img - mask_center_y + j_mask)) * mask[i_mask][j_mask]
                    sum_r += value_r
                    
                    value_g = g.getpixel((i_img - mask_center_x + i_mask, j_img - mask_center_y + j_mask)) * mask[i_mask][j_mask]
                    sum_g += value_g
                    
                    value_b = b.getpixel((i_img - mask_center_x + i_mask, j_img - mask_center_y + j_mask)) * mask[i_mask][j_mask]
                    sum_b += value_b

            output[row, col] = [
                min(max(int(sum_r), 0), 255) if activation_function == 'relu' else sum_r,
                min(max(int(sum_g), 0), 255) if activation_function == 'relu' else sum_g,
                min(max(int(sum_b), 0), 255) if activation_function == 'relu' else sum_b,
            ]
            col += 1
        row += 1
    
    im_result = Image.fromarray(output, mode='RGB')

    return im_result