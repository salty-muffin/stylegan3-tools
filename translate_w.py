import click
import numpy as np


# fmt: off
@click.command()
@click.option("--start_w", type=click.Path(exists=True, dir_okay=False),       help="the starting vector w to be translated (as an .npz file)", required=True)
@click.option("--translation_w", type=click.Path(exists=True, dir_okay=False), help="the vector for translation (as an .npz / .npy file)", required=True)
@click.option("--factor", type=float, default=0.0,                             help="the factor for translation", required=True)
@click.option("--outfile", type=click.Path(dir_okay=False),                    help="where to save the translated vector (as an .npz file)", required=True)
# fmt: on
def translate_w(start_w: str, translation_w: str, factor: float, outfile: str):
    # load vectors
    start = np.load(start_w)["w"]
    translation = np.load(translation_w)
    if ".npz" in translation_w:
        translation = translation["w"]

    # calculate translation
    translated = start + factor * translation

    # save result
    np.savez(outfile, w=translated)


if __name__ == "__main__":
    translate_w()
