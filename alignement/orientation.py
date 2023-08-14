from typing import Tuple

import numpy as np
import math
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont


def visualize_orientation(
    filepath: str,
    eye_left: np.ndarray,
    eye_right: np.ndarray,
    mouth: np.ndarray,
    quad: np.ndarray,
    pitch=0.0,
    yaw=0.0,
    roll=0.0,
) -> PIL.Image.Image:
    # read image
    img = PIL.Image.open(filepath)

    # draw lines
    draw = PIL.ImageDraw.Draw(img)
    for index, point in enumerate(quad):
        draw.line([tuple(point), tuple(quad[(index + 1) % len(quad)])], "#ff0000", 3)

    draw.line([tuple(eye_left), tuple(eye_right)], "#0000ff", 3)
    draw.line([tuple(eye_left), tuple(mouth)], "#00ff00", 3)
    draw.line([tuple(mouth), tuple(eye_right)], "#00ff00", 3)
    draw.line(
        [tuple(np.mean((eye_left, eye_right), axis=0)), tuple(mouth)], "#0000ff", 3
    )

    # draw text
    draw.multiline_text(
        (10, 10), f"pitch: {pitch}\nyaw: {yaw}\nroll: {roll}", "#ffffff"
    )

    return img.resize((1024, 1024), PIL.Image.ANTIALIAS)


def calculate_orientation(
    eye_left: np.ndarray,
    eye_right: np.ndarray,
    mouth: np.ndarray,
    ref_eye_left: np.ndarray,
    ref_eye_right: np.ndarray,
    ref_mouth: np.ndarray,
) -> Tuple[float]:
    ref_roll_y = (
        (ref_eye_right - ref_eye_left) / np.linalg.norm(ref_eye_right - ref_eye_left)
    )[1]
    ref_roll = math.acos(
        np.clip(
            ref_roll_y,
            0,
            1,
        )
    ) * (1 if ref_roll_y > 0 else -1)
    roll_y = ((eye_right - eye_left) / np.linalg.norm(eye_right - eye_left))[1]
    roll = math.acos(np.clip(roll_y, 0, 1)) * (1 if roll_y > 0 else -1) - ref_roll

    ref_eye_to_eye_len = np.linalg.norm(ref_eye_right - ref_eye_left, axis=0)
    eye_to_eye_len = np.linalg.norm(eye_right - eye_left, axis=0)
    yaw = math.acos(np.clip(eye_to_eye_len / ref_eye_to_eye_len, 0, 1))

    ref_eye_to_mouth_len = np.linalg.norm(
        ref_mouth - np.mean((ref_eye_left, ref_eye_right), axis=0)
    )
    eye_to_mouth_len = np.linalg.norm(mouth - np.mean((eye_left, eye_right), axis=0))
    pitch = math.acos(np.clip(eye_to_mouth_len / ref_eye_to_mouth_len, 0, 1))

    return pitch, yaw, roll
