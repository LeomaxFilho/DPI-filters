import json
from pathlib import Path

import numpy as np
from PIL import Image


def atrous_dilatation(
    dilatation_rate: int, mask: list[list[int]], mask_size: tuple[int, int]
):
    "faz a dilatacao da mascara de entrada"
    if dilatation_rate <= 1:
        return mask

    Rows, Columns = mask_size
    mask_dilated: list[list[int]] = []
    zeros = [0 for i in range(Rows + dilatation_rate)]

    for row in range(Rows):
        new_row = []

        for column in range(Columns):
            new_row.append(mask[row][column])

            if column + 1 != Columns:
                for r in range(dilatation_rate - 1):
                    new_row.append(0)

        mask_dilated.append(new_row)

        if row + 1 != Rows:
            for r in range(dilatation_rate - 1):
                mask_dilated.append(zeros)

    return mask_dilated


def read_json(path: Path):
    """ler json"""

    with open(path, "r") as file:
        options = json.load(file)

    dilatation_rate = options["dilatation_rate"]
    mask_dilated = atrous_dilatation(
        dilatation_rate,
        options["mask"],
        (len(options["mask"]), len(options["mask"][0])),
    )
    mask = np.array(mask_dilated)
    mask_size = (len(mask), len(mask[0]))
    function = options["function"].lower()
    stride = options["stride"]
    activation_function = options["activation_function"].lower()

    return mask, mask_size, function, stride, dilatation_rate, activation_function


def read_image(path: Path):
    "ler imagem"
    im = Image.open(path)
    band = im.split()
    r, g, b = band[:3]

    a = band[3] if len(band) == 4 else None

    im_size = im.size

    return r, g, b, im, im_size, a


def save_image(r, g, b, path: Path):
    "salvar imagem"
    im = Image.merge("RGB", (r, g, b))
    im.save(path)


def rgb2hex(colour: tuple[int, int, int]):
    return "#{:02x}{:02x}{:02x}".format(colour[0], colour[1], colour[2])
