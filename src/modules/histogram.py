from PIL import Image
import numpy as np

def histogram_exp(image : Image.Image):
    band = image.split()
    has_alpha = len(band) == 4

    if has_alpha:
        r, g, b, a = band
    else:
        r, g, b = band
        a = None

    r = histogram_exp_band(r)
    g = histogram_exp_band(g)
    b = histogram_exp_band(b)
    #a = histogram_exp_band(a) if has_alpha else None
    
    bands = np.array((r, g, b))# if not has_alpha else np.array((r, g, b, a))

    #if not has_alpha:
    bands = np.stack((np.asarray(r), np.asarray(g), np.asarray(b)), axis=-1)
    #else:
        #bands = np.stack((np.asarray(r), np.asarray(g), np.asarray(b), np.asarray(a)), axis=-1)

    im_result = Image.fromarray(bands, mode="RGB") #if not a else "RGBA")
    
    return im_result

def histogram_exp_band(band : Image.Image):
    band_arr = np.array(list(map(np.abs, np.asarray(band))))
    r_max, r_min = band_arr.max() , band_arr.min()
    L = 256

    #print(r_max, r_min)
    #if r_max == r_min:
        #return Image.fromarray(band_arr.astype(np.uint8))
    
    band_result = np.rint(((band_arr - r_min) / (r_max - r_min))* (L - 1)).astype(np.uint8)

    return Image.fromarray(band_result)
