from modules.correlation import to_img
import numpy as np

def histogram_exp(image_arr : np.ndarray):
    if len(image_arr[0]) <= 3:
        r, g, b = np.hsplit(image_arr, 3)
        a = None
    else:
        r, g, b, a = np.hsplit(image_arr, 4)

    r = histogram_exp_band(r)
    g = histogram_exp_band(g)
    b = histogram_exp_band(b)
    
    bands = np.stack((r, g, b), -1) if not a else np.stack((r, g, b, a), -1)

    im_result = to_img(bands)
    
    return im_result

def histogram_exp_band(band : np.ndarray) -> np.ndarray:
    band_arr = np.array(list(map(np.abs, band)))
    r_max, r_min = band_arr.max() , band_arr.min()
    L = 256
    
    band_result : np.ndarray = np.rint(((band_arr - r_min) / (r_max - r_min))* (L - 1))

    return band_result
