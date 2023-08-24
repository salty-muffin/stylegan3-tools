from typing import List, Union
import os
import click
import numpy as np
from tqdm import tqdm

from gen_images import parse_paths


def parse_keyframes(s: str) -> List[tuple[int, float]]:
    keyframes = []
    for kf in s.split(","):
        l = kf.strip().split(":")
        keyframes.append((int(l[0].strip()), float(l[1].strip())))

    return keyframes


def get_keyframes(
    current: int, kfs: List[tuple[int, float]]
) -> tuple[Union[tuple[int, float], None, float]]:
    prev = None
    next = None

    prev_i = 0
    for i, kf in enumerate(kfs):
        if kf[0] == current:
            prev = kf
            prev_i = i
            break
        if kf[0] > current:
            if i - 1 >= 0:
                prev = kfs[i - 1]
                prev_i = i - 1
            break

    if prev is None:
        next = kfs[0]
    elif prev_i + 1 < len(kfs):
        next = kfs[prev_i + 1]

    factor = 0
    if prev is None:
        factor = 1
    elif next is not None:
        factor = (current - prev[0]) / (next[0] - prev[0])

    return (
        prev[1] if prev is not None else None,
        next[1] if next is not None else None,
        factor,
    )


def linear_interp(a: float, b: float, t: float) -> float:
    return a * (1 - t) + b * t


# fmt: off
@click.command()
@click.option("--start_ws",       type=parse_paths,                             help="the starting vector w to be translated (as an .npz file)", required=True)
@click.option("--translation_w",  type=click.Path(exists=True, dir_okay=False), help="the vector for translation (as an .npz / .npy file)", required=True)
@click.option("--keyframes",      type=parse_keyframes,                         help="list of keyframes to interpolate between (format: '0:0.0,5:1.0,8:0.1')", required=True)
@click.option("--outdir",         type=click.Path(file_okay=False),             help="where to save the translated vectors (as .npz files)", required=True)
# fmt: on
def batch_translate_w(
    start_ws: str, translation_w: str, keyframes: List[tuple[int, float]], outdir: str
):
    os.makedirs(outdir, exist_ok=True)

    # load vectors
    translation = np.load(translation_w)
    if ".npz" in translation_w:
        translation = translation["w"]

    if len(keyframes):
        for index, start_w in enumerate(tqdm(start_ws)):
            # load vectors
            start = np.load(start_w)["w"]

            prev, next, factor = get_keyframes(index, keyframes)

            # calculate translation
            translated = start
            if prev is None:
                translated += translation * next
            elif next is None:
                translated += translation * prev
            else:
                translated += translation * linear_interp(prev, next, factor)

            # save result
            np.savez(os.path.join(outdir, os.path.basename(start_w)), w=translated)


if __name__ == "__main__":
    batch_translate_w()
