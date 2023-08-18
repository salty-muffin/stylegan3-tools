"""
tests if all images in in a directory can be read.
"""

import os
from tqdm import tqdm
from glob import glob
import click
import PIL.Image
from pathlib import Path

from align import file_ext, is_image_ext, error


# fmt: off
@click.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False), metavar="DIR")
# fmt: on
def verify(directory: str) -> None:
    PIL.Image.init()

    if not len(
        matches := [
            str(f)
            for f in sorted(Path(directory).glob("*"))
            if is_image_ext(f) and os.path.isfile(f)
        ]
    ):
        error(f"no compatible images found in {directory}")

    default_ext = file_ext(matches[0])

    img = PIL.Image.open(matches[0])
    default_size = img.size

    if len(matches) > 1:
        for path in tqdm(matches):
            try:
                ext = file_ext(path)
                img = PIL.Image.open(path)
                size = img.size
                if ext != default_ext:
                    print(
                        f"'{path}' diverges from extension: '{ext}', default: '{default_ext}"
                    )
                if size != default_size:
                    print(
                        f"'{path}' diverges from size: '{size}', default: '{default_size}"
                    )
            except Exception:
                print(f"could not open '{path}' as image")


if __name__ == "__main__":
    verify()
