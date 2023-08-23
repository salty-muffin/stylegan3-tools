"""
tool to align faces similarly to the ffhq dataset. These can then be used for transfer learning and projection

this tool aditionally calculates the head orientation and saves this and the crop rectangle dimensions.
"""

from typing import Union, Optional

import os
import sys
import click
import numpy as np
from pathlib import Path
import PIL.Image
import dlib
import imageio
import yaml

from alignement.align_face import get_landmarks, align_face
from alignement.orientation import visualize_orientation, calculate_orientation

# ----------------------------------------------------------------------------


def error(msg):
    print("Error: " + msg)
    sys.exit(1)


# ----------------------------------------------------------------------------


def file_ext(name: Union[str, Path]) -> str:
    return str(name).split(".")[-1]


# ----------------------------------------------------------------------------


def is_image_ext(fname: Union[str, Path]) -> bool:
    ext = file_ext(fname).lower()
    return f".{ext}" in PIL.Image.EXTENSION  # type: ignore


# fmt: off
@click.command()
@click.option("--predictor", "predictor_dat",    type=click.Path(exists=True, dir_okay=False), help="Dlib landmark detection model filename (.dat)", required=True, metavar="PATH")
@click.option("--source",                        type=click.Path(exists=True, file_okay=False), help="Directory for input images", required=True, metavar="DIR")
@click.option("--dest",                          type=click.Path(file_okay=False), help="Output directory for aligned images", required=True, metavar="DIR")
@click.option("--data-dest", "data_dest",        type=click.Path(dir_okay=False), help=".yml file to output the angle & crop data to (if not set, then no extra data will be saved)", required=False, metavar="PATH")
@click.option("--fps",                           type=click.IntRange(1, 60), help="fps of the output video", default=30)
@click.option("--video-dest", "video_dest",      type=click.Path(dir_okay=False), help="output path for a video file of orientation markers (if desired)", required=False, metavar="PATH")
@click.option("--landmarker", "landmarker_task", type=click.Path(exists=True, dir_okay=False), help="Mediapipe landmark detection model filename (.task). If this is not set, no orientation data will be saved", required=False, metavar="PATH")
# fmt: on
def run_alignment(
    predictor_dat: str,
    source: str,
    dest: str,
    data_dest: str,
    fps: int,
    video_dest: Optional[str],
    landmarker_task: Optional[str],
):
    PIL.Image.init()

    # create output directories, if it does not exists
    os.makedirs(dest, exist_ok=True)
    os.makedirs(os.path.split(data_dest)[0], exist_ok=True)
    os.makedirs(os.path.split(video_dest)[0], exist_ok=True)

    # find all images in source directory
    if not len(
        matches := [
            str(f)
            for f in sorted(Path(source).rglob("*"))
            if is_image_ext(f) and os.path.isfile(f)
        ]
    ):
        error(f"no compatible images found in {source}")

    # load models for getting landmarks
    predictor = dlib.shape_predictor(predictor_dat)

    video = None
    if video_dest and data_dest:
        video = imageio.get_writer(
            video_dest, mode="I", fps=fps, codec="libx264", bitrate="16M"
        )
    # align faces and save files for the number of matches
    data = []
    try:
        for path in matches:
            landmarks = get_landmarks(path, predictor)
            alignments = align_face(path, landmarks)
            if alignments is not None:

                def get_orientation(alignment, outpath):
                    lm_result = None
                    if landmarker_task:
                        pitch, yaw, roll, x, y, z, lm_result = calculate_orientation(
                            path, landmarker_task
                        )

                        data.append(
                            {
                                "path": os.path.split(outpath)[1],
                                "pitch": float(pitch),
                                "yaw": float(yaw),
                                "roll": float(roll),
                                "x": float(x),
                                "y": float(y),
                                "z": float(z),
                                "quad": [
                                    {"x": float(point[0]), "y": float(point[1])}
                                    for point in alignment["quad"]
                                ],
                            }
                        )
                    else:
                        data.append(
                            {
                                "path": os.path.split(outpath)[1],
                                "quad": [
                                    {"x": float(point[0]), "y": float(point[1])}
                                    for point in alignment["quad"]
                                ],
                            }
                        )

                    with open(data_dest, "w+") as file:
                        yaml.dump(data, file)

                    if video_dest:
                        video.append_data(
                            np.array(
                                visualize_orientation(
                                    path,
                                    alignment["quad"],
                                    pitch,
                                    yaw,
                                    roll,
                                    x,
                                    y,
                                    z,
                                    lm_result,
                                )
                            )
                        )

                if len(alignments) == 1:
                    outpath = os.path.join(
                        dest,
                        os.path.basename(path).replace(
                            file_ext(os.path.basename(path)), "png"
                        ),
                    )
                    alignments[0]["image"].save(outpath)
                    if data_dest:
                        get_orientation(alignments[0], outpath)

                elif len(alignments) > 1:
                    for index, alignment in enumerate(alignments):
                        name, extension = os.path.splitext(os.path.basename(path))
                        outpath = os.path.join(dest, f"{name}_{index:02d}.png")
                        alignment["image"].save(outpath)
                        if data_dest:
                            get_orientation(alignment, outpath)
                elif video_dest and data_dest:
                    video.append_data(np.array(PIL.Image.new("RGB", (1024, 1024))))
    finally:
        if video_dest and data_dest:
            video.close()


if __name__ == "__main__":
    run_alignment()  # pylint: disable=no-value-for-parameter
