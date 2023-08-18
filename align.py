"""
tool to align faces similarly to the ffhq dataset. These can then be used for transfer learning and projection

this tool aditionally calculates the head orientation and saves this and the crop rectangle dimensions.
"""

from typing import Union

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
@click.option("--predictor", "predictor_dat", type=click.Path(exists=True, dir_okay=False), help="Landmark detection model filename", required=True, metavar="PATH")
@click.option("--source",                     type=click.Path(exists=True, file_okay=False), help="Directory for input images", required=True, metavar="DIR")
@click.option("--dest",                       type=click.Path(file_okay=False), help="Output directory for aligned images", required=True, metavar="DIR")
@click.option("--base-image", "base_image",   type=click.Path(exists=True, dir_okay=False), help="Calibration image (preferably with neutral head orientation)", required=False, metavar="PATH")
@click.option("--data-dest", "data_dest",     type=click.Path(dir_okay=False), help=".yml file to output the angle & crop data to (if not set, then no extra data will be saved)", required=False, metavar="PATH")
@click.option("--video-dest", "video_dest",   type=click.Path(dir_okay=False), help="output path for a video file of orientation markers (if desired)", required=False, metavar="PATH")
@click.option("--fps",                        type=click.IntRange(1, 60), help="fps of the output video", default=30)
# fmt: on
def run_alignment(
    predictor_dat: str,
    source: str,
    dest: str,
    base_image: str,
    data_dest: str,
    video_dest: str,
    fps: int,
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

    # args check
    if base_image and not data_dest:
        print(
            "warning: base-image was set, but no destination for the calculated data was given. no meta data will be saved"
        )
    if data_dest and not base_image:
        print(
            "warning: destination for the meta data was given, but no calibration image. the first image will be used instead"
        )
        base_image = matches[0]

    # load predictor for getting landmarks
    predictor = dlib.shape_predictor(predictor_dat)

    # get reference image
    ref_landmarks = get_landmarks(base_image, predictor)
    ref_alignments = align_face(base_image, ref_landmarks)

    if not len(ref_alignments) and data_dest:
        error("no alignemnts for the reference image could be generated. aborting")

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

                def get_orientation(alignment, ref_alignment, outpath):
                    pitch, yaw, roll, nose_direction = calculate_orientation(
                        alignment["eye_left"],
                        alignment["eye_right"],
                        alignment["mouth_avg"],
                        alignment["nose_tip"],
                        ref_alignment["eye_left"],
                        ref_alignment["eye_right"],
                        ref_alignment["mouth_avg"],
                        ref_alignment["nose_tip"],
                    )

                    data.append(
                        {
                            "path": os.path.split(outpath)[1],
                            "pitch": float(pitch),
                            "yaw": float(yaw),
                            "roll": float(roll),
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
                                    alignment["eye_left"],
                                    alignment["eye_right"],
                                    alignment["mouth_avg"],
                                    alignment["quad"],
                                    alignment["lm"],
                                    pitch,
                                    yaw,
                                    roll,
                                    alignment["nose_tip"],
                                    nose_direction,
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
                        get_orientation(alignments[0], ref_alignments[0], outpath)

                elif len(alignments) > 1:
                    for index, alignment in enumerate(alignments):
                        name, extension = os.path.splitext(os.path.basename(path))
                        outpath = os.path.join(dest, f"{name}_{index:02d}.png")
                        alignment["image"].save(outpath)
                        if data_dest:
                            get_orientation(alignment, ref_alignments[0], outpath)
                elif video_dest and data_dest:
                    video.append_data(np.array(PIL.Image.new("RGB", (1024, 1024))))
    finally:
        if video_dest and data_dest:
            video.close()


if __name__ == "__main__":
    run_alignment()  # pylint: disable=no-value-for-parameter
