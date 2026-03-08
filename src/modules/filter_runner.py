from pathlib import Path

from modules.correlation import correlation
from modules.read_save import read_image, read_json
from modules.sobel import sobel

import os
import sys


def apply_filter(image_path: Path, filter_path: Path):
    mask, mask_size, function, stride, dilatation_rate, activation_function = read_json(filter_path)
    r, g, b, _, im_size, a = read_image(image_path)

    if function == "sobel":
        return sobel(r,g,b,im_size,mask,mask_size,stride,activation_function,a=a,)
    else:
        return correlation(r,g,b,im_size,mask,mask_size,stride,activation_function,a=a,)


def run_filter_pipeline():

    path_options = Path(os.getcwd()).parent / "data" / "filter" / "filter.json"
    path_image = Path(os.getcwd()).parent / "data" / "images" / "Shapes.png"
    path_image_apple = Path(os.getcwd()).parent / "data" / "images" / "apple_img.png"
    path_image_2 = Path(os.getcwd()).parent / "data" / "images" / "testpat.1k.color2.tif"
    path_output = Path(os.getcwd()).parent / "data" / "images" / "output.png"
    
    result_image = apply_filter(path_image, path_options)

    result_image.save(path_output)

    return path_output
