import json
from path import Path
from PIL import Image

def read_json(path : Path):
    """ler json"""

    with open(path, 'r') as file:
        options = json.load(file)

    mask = options['mask']
    mask_size = (len(mask), len(mask[0]))
    function = options['function'].lower()
    stride = options['stride']
    dilatation_rate = options['dilatation_rate']
    activation_function = options['activation_function'].lower()

    return mask, mask_size, function, stride, dilatation_rate, activation_function

def read_image(path : Path):
    'ler imagem'
    im = Image.open(path)
    r, g, b = im.split()
    x_size, y_size = im.size

    return r, g, b, im, x_size, y_size

def save_image(r, g, b, path : Path):
    'salvar imagem'
    im = Image.merge('RGB', (r, g, b))
    im.save(path)