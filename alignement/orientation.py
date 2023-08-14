import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont


def visualize_orientation(
    filepath: str,
    eye_left: np.ndarray,
    eye_right: np.ndarray,
    mouth: np.ndarray,
    quad: np.ndarray,
    pitch=0,
    yaw=0,
    roll=0,
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
