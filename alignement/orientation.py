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
    lm=[],
    pitch=0.0,
    yaw=0.0,
    roll=0.0,
    nose_tip=np.array((0.0, 0.0)),
    nose_direction=np.array((0.0, 0.0)),
) -> PIL.Image.Image:
    # read image
    img = PIL.Image.open(filepath)

    img = img.resize((1024, 1024), PIL.Image.ANTIALIAS)

    # draw lines
    draw = PIL.ImageDraw.Draw(img)
    for index, point in enumerate(quad * 1024):
        draw.line(
            [tuple(point), tuple(quad[(index + 1) % len(quad)] * 1024)], "#ff0000", 3
        )

    draw.line([tuple(eye_left * 1024), tuple(eye_right * 1024)], "#ffffff", 3)
    draw.line([tuple(eye_left * 1024), tuple(mouth * 1024)], "#00ff00", 3)
    draw.line([tuple(mouth * 1024), tuple(eye_right * 1024)], "#00ff00", 3)
    draw.line(
        [tuple(np.mean((eye_left, eye_right), axis=0) * 1024), tuple(mouth * 1024)],
        "#ffffff",
        3,
    )

    draw.line(
        [tuple(nose_tip * 1024), tuple((nose_tip + nose_direction) * 1024)],
        "#0000ff",
        3,
    )

    for point in lm * 1024:
        draw.line(
            [tuple(point + np.array((-1.5, 0))), tuple(point + np.array((1.5, 0)))],
            "#0000ff",
            3,
        )

    # draw text
    draw.multiline_text(
        (10, 10), f"pitch: {pitch}°\nyaw: {yaw}°\nroll: {roll}°", "#ffffff"
    )

    return img


def calculate_orientation(
    eye_left: np.ndarray,
    eye_right: np.ndarray,
    mouth: np.ndarray,
    nose_tip: np.ndarray,
    ref_eye_left: np.ndarray,
    ref_eye_right: np.ndarray,
    ref_mouth: np.ndarray,
    ref_nose_tip: np.ndarray,
) -> Tuple[float]:
    # calculate roll
    ref_roll_y = (
        (ref_eye_right - ref_eye_left) / np.hypot(*(ref_eye_right - ref_eye_left))
    )[1]
    ref_roll = math.acos(
        np.clip(
            ref_roll_y,
            0,
            1,
        )
    ) * (1 if ref_roll_y > 0 else -1)
    roll_y = ((eye_right - eye_left) / np.hypot(*(eye_right - eye_left)))[1]
    roll = math.acos(np.clip(roll_y, 0, 1)) * (-1 if roll_y < 0 else 1) - ref_roll

    # get neutral nose tip direction
    ref_eye_average = np.mean((ref_eye_left, ref_eye_right), axis=0)
    ref_eye_to_mouth = ref_mouth - ref_eye_average
    ref_eye_to_mouth_norm = ref_eye_to_mouth / np.hypot(*ref_eye_to_mouth)
    ref_v = ref_nose_tip - ref_eye_average
    ref_d = np.dot(ref_v, ref_eye_to_mouth_norm)
    ref_nose_base = ref_eye_average + ref_eye_to_mouth_norm * ref_d
    ref_nose_base_to_tip = ref_nose_tip - ref_nose_base

    # get current nose tip direction
    eye_average = np.mean((eye_left, eye_right), axis=0)
    eye_to_mouth = mouth - eye_average
    eye_to_mouth_norm = eye_to_mouth / np.hypot(*eye_to_mouth)
    nose_base = eye_average + eye_to_mouth_norm * ref_d
    nose_base_to_tip = nose_tip - nose_base

    nose_direction = nose_base_to_tip - ref_nose_base_to_tip

    # calculate yaw
    ref_eye_to_eye_len = np.hypot(*(ref_eye_right - ref_eye_left))
    eye_to_eye_len = np.hypot(*(eye_right - eye_left))
    yaw = math.acos(np.clip(eye_to_eye_len / ref_eye_to_eye_len, 0, 1)) * (
        -1 if nose_direction[0] < 0 else 1
    )

    # calculate pitch
    ref_eye_to_mouth_len = np.hypot(
        *(ref_mouth - np.mean((ref_eye_left, ref_eye_right), axis=0))
    )
    eye_to_mouth_len = np.hypot(*(mouth - np.mean((eye_left, eye_right), axis=0)))
    pitch = math.acos(np.clip(eye_to_mouth_len / ref_eye_to_mouth_len, 0, 1)) * (
        -1 if nose_direction[1] < 0 else 1
    )

    return np.rad2deg(pitch), np.rad2deg(yaw), np.rad2deg(roll), nose_direction
