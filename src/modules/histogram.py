from PIL import Image
import numpy as np

def histogram_exp(image : Image.Image):
    band = image.split()
    r, g, b = band

    a = (band[3] if len(band) == 4 else None)

    r = histogram_exp_band(r)
    g = histogram_exp_band(g)
    b = histogram_exp_band(b)
    a = histogram_exp_band(a) if a else None
    
    bands = np.array((r, g, b)) if not a else np.array((r, g, b, a))

    im_result = Image.fromarray(bands, mode="RGB" if not a else "RGBA")
    
    return im_result

def histogram_exp_band(band : Image.Image):
    band_arr = np.array(list(map(np.abs, np.asarray(band))))
    r_max, r_min = band_arr.max() , band_arr.min()
    L = 256

    band_result = np.array(list(map(lambda r : round((r - r_min)/(r_max - r_min) * (L - 1)), band_arr)))

    return Image.fromarray(band_result)
