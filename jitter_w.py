from typing import List, Union
import os
import glob
import click
import random
import numpy as np
from tqdm import tqdm

from gen_images import parse_paths


def parse_float_comma_list(s: str) -> List[float]:
    return [float(s.strip()) for s in s.split(",")]


def parse_npy_paths(s: Union[str, List]) -> List[int]:
    if isinstance(s, list):
        return s
    filenames = []
    if os.path.isdir(s):
        filenames = glob.glob(os.path.join(s, "*.npy"))
        filenames.sort()
    else:
        for p in s.split(","):
            filenames.append(p.strip())
    return filenames


# fmt: off
@click.command()
@click.option("--start_ws",       type=parse_paths,                                 help="the starting vector w to be translated (as an .npz file)", required=True)
@click.option("--translation_ws", type=parse_npy_paths,                             help="the vectors for translation (as .npz / .npy files)", required=True)
@click.option("--magnitude",      type=parse_float_comma_list, default=[-1.0, 1.0], help="the factor for translation", required=True)
@click.option("--outdir",         type=click.Path(file_okay=False),                 help="where to save the translated vector(s) (as an .npz file)", required=True)
# fmt: on
def jitter_w(start_ws: str, translation_ws: str, magnitude: List[float], outdir: str):
    os.makedirs(outdir, exist_ok=True)

    # load vectors
    for start_w in tqdm(start_ws):
        vector = np.load(start_w)["w"]

        for translation_w in translation_ws:
            translation = np.load(translation_w)

            # calculate translation
            vector += random.uniform(magnitude[0], magnitude[-1]) * translation

        # save result
        np.savez(os.path.join(outdir, os.path.basename(start_w)), w=vector)


if __name__ == "__main__":
    jitter_w()
