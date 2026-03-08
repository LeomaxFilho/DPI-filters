from pathlib import Path

from modules.correlation import correlation, to_img
from modules.read_save import read_image, read_json
from modules.sobel import full_axis_sobel, sobel



def apply_filter(image_path: Path, filter_path: Path):
    mask, mask_size, function, stride, _, activation_function = read_json(filter_path)
    r, g, b, _, im_size, a = read_image(image_path)

    if function == "sobel":
        return full_axis_sobel(r, g, b, im_size, mask, mask_size, stride, activation_function, a=a)
    elif function == "full_sobel":
        return sobel(r, g, b, im_size, mask, mask_size, stride, activation_function, a=a)
    else:
        im_arr = correlation(r,g,b,im_size,mask,mask_size,stride,activation_function,a=a)
        return to_img(im_arr)

def run_filter_pipeline():

    path_options = Path(__file__).parent.parent.parent / "data" / "filter" / "filter.json"
    path_image = Path(__file__).parent.parent.parent / "data" / "images" / "Shapes.png"
    path_image_apple = Path(__file__).parent.parent.parent / "data" / "images" / "apple_img.png"
    path_image_2 = Path(__file__).parent.parent.parent / "data" / "images" / "testpat.1k.color2.tif"
    path_output = Path(__file__).parent.parent.parent / "data" / "images" / "output.png"
    
    result_image = apply_filter(path_image_2, path_options)

    result_image.save(path_output)

    return path_output
