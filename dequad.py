"""
resets an aligned image of the face into the position it was pre alignment
"""

from typing import Tuple

import os
import click
import yaml
from glob import glob
from itertools import chain
from wand.color import Color
from wand.image import Image


def parse_dimensions(s: str) -> Tuple[int]:
    return tuple(int(dim.strip()) for dim in s.split("x"))


# fmt: off
@click.command()
@click.option("--source", type=click.Path(exists=True, file_okay=False), help="Directory for input images", required=True, metavar="PATH")
@click.option("--dest",   type=click.Path(file_okay=False), help="Output directory for aligned images", required=True, metavar="PATH")
@click.option("--data",   type=click.Path(exists=True, dir_okay=False), help=".yml file to output the angle & crop data to (if not set, then no extra data will be saved)", required=False, metavar="PATH")
@click.option("--size",   type=parse_dimensions, help="output size (width & height, e.g. 1024x1024)", default=(1024, 1024))
# fmt: on
def dequad_images(source: str, dest: str, data: str, size: Tuple[int]) -> None:
    os.makedirs(dest, exist_ok=True)

    target_fnames = sorted(glob(os.path.join(source, "*.png")))

    with open(data) as file:
        metadata = yaml.safe_load(file)


if __name__ == "__main__":
    dequad_images()
