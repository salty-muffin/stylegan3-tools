"""
resets an aligned image of the face into the position it was pre alignment
"""

import os
import click
import yaml
from glob import glob
import numpy as np
from itertools import chain
from wand.image import Image
from tqdm import tqdm


def parse_dimensions(s: str) -> tuple[int]:
    return tuple(int(dim.strip()) for dim in s.split("x"))


# fmt: off
@click.command()
@click.option("--source", type=click.Path(exists=True, file_okay=False), help="Directory for input images", required=True, metavar="PATH")
@click.option("--dest",   type=click.Path(file_okay=False), help="Output directory for aligned images", required=True, metavar="PATH")
@click.option("--data",   type=click.Path(exists=True, dir_okay=False), help=".yml file to output the angle & crop data to (if not set, then no extra data will be saved)", required=False, metavar="PATH")
@click.option("--size",   type=parse_dimensions, help="output size (width & height, e.g. 1024x1024)", default=(1024, 1024))
# fmt: on
def dequad_images(source: str, dest: str, data: str, size: tuple[int]) -> None:
    # create output directory
    os.makedirs(dest, exist_ok=True)

    # get all images from source directory
    target_fnames = sorted(glob(os.path.join(source, "*.png")))

    # load metadata
    with open(data) as file:
        metadata = yaml.safe_load(file)

    size = np.array(size)

    corners = (
        (0, 0),
        (0, size[1]),
        tuple(size),
        (size[0], 0),
    )

    for path in tqdm(target_fnames):
        filtered = list(filter(lambda x: x["path"] == os.path.basename(path), metadata))

        if len(filtered):
            quad = np.array([(p["x"], p["y"]) for p in filtered[0]["quad"]])
            quad = quad * size

            with Image(filename=path) as img:
                img.resize(*tuple(size.astype("int32")), "mitchell")
                img.virtual_pixel = "transparent"

                order = chain.from_iterable(
                    zip(corners, [tuple(point) for point in quad])
                )
                arguments = list(chain.from_iterable(order))
                img.distort(
                    "perspective",
                    arguments,
                )
                img.save(filename=os.path.join(dest, os.path.basename(path)))
        else:
            print(f"warning: no match for '{path}' could be found in '{data}'")


if __name__ == "__main__":
    dequad_images()
