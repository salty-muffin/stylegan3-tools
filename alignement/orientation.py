from typing import Tuple, Any

import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from scipy.spatial.transform import Rotation as R


def draw_landmarks_on_image(
    image: PIL.Image.Image, detection_result
) -> PIL.Image.Image:
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.array(image.convert("RGB"))

    # Loop through the detected faces to visualize.
    for face_landmarks in face_landmarks_list:
        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in face_landmarks
            ]
        )

        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style(),
        )
        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style(),
        )

    return PIL.Image.fromarray(annotated_image).convert(image.mode)


def visualize_orientation(
    filepath: str,
    quad: np.ndarray,
    pitch=0.0,
    yaw=0.0,
    roll=0.0,
    x=0.0,
    y=0.0,
    z=0.0,
    lm_result=None,
) -> PIL.Image.Image:
    # read image
    img = PIL.Image.open(filepath)

    img = img.resize((1024, 1024), PIL.Image.ANTIALIAS)

    # draw quad lines
    draw = PIL.ImageDraw.Draw(img)
    for index, point in enumerate(quad * 1024):
        draw.line(
            [tuple(point), tuple(quad[(index + 1) % len(quad)] * 1024)], "#ff0000", 3
        )

    # draw text
    draw.multiline_text(
        (10, 10),
        f"pitch: {pitch}°\nyaw: {yaw}°\nroll: {roll}°\nx: {x}\ny: {y}\n z: {z}",
        "#ffffff",
    )

    # draw face mesh
    if lm_result:
        img = draw_landmarks_on_image(img, lm_result)

    return img


def calculate_orientation(filepath: str, landmarker_path: str) -> Tuple[float, Any]:
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=landmarker_path),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )

    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image.create_from_file(filepath)

        face_landmarker_result = landmarker.detect(mp_image)

        matrix = np.array(
            face_landmarker_result.facial_transformation_matrixes
        ).reshape((4, 4))

        r = R.from_matrix(matrix[:3, :3])

        rotvec = r.as_rotvec(degrees=True)

        return (
            rotvec[0],
            rotvec[1],
            rotvec[2],
            matrix[0, 3],
            matrix[1, 3],
            matrix[2, 3],
            face_landmarker_result,
        )
