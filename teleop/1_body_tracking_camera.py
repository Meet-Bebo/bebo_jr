"""Track a human pose from a live camera feed and stream it to Placo."""

from __future__ import annotations

import argparse
import cv2
import mediapipe as mp

from placo_utils.visualization import point_viz, points_viz

from body_tracking_common import (
    Point,
    DEFAULT_VISUALIZATION_POINTS,
    get_transformation_matrix,
    points_from_landmarks,
    select_named_points,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Index of the OpenCV camera to use.",
    )
    parser.add_argument(
        "--shoulder-width",
        type=float,
        default=0.178,
        help="Desired shoulder width of the normalized skeleton (meters).",
    )
    parser.add_argument(
        "--shoulder-height",
        type=float,
        default=0.179517,
        help="Height (meters) to place the normalized shoulder midpoint.",
    )
    return parser.parse_args()


def track_from_camera(camera_index: int, shoulder_width: float, shoulder_height: float) -> None:
    """Continuously capture frames, estimate pose, and publish landmarks."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {camera_index}")

    mp_pose = mp.solutions.pose

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Failed to read frame from camera.")
                    break

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)
                points = points_from_landmarks(results.pose_landmarks)
                if not points:
                    continue

                transform = get_transformation_matrix(points, shoulder_width, shoulder_height)
                transformed_points = [point.transform(transform) for point in points]

                named_points = select_named_points(transformed_points, DEFAULT_VISUALIZATION_POINTS)

                if named_points:
                    publish_points(named_points)

        except KeyboardInterrupt:
            print("Stopping camera tracking.")
        finally:
            cap.release()

def publish_points(points: dict[str, Point]) -> None:
    """Push the selected points to the Placo visualizer."""
    right_points = [
        points["RIGHT_ELBOW"].xyz,
        points["RIGHT_SHOULDER"].xyz,
        points["RIGHT_WRIST"].xyz,
    ]
    left_points = [
        points["LEFT_ELBOW"].xyz,
        points["LEFT_SHOULDER"].xyz,
        points["LEFT_WRIST"].xyz,
    ]

    points_viz("points", right_points, radius=0.05, color=0x0000FF)
    points_viz("pointss", left_points, radius=0.05, color=0xFF0000)


if __name__ == "__main__":
    args = parse_args()
    track_from_camera(
        camera_index=args.camera_index,
        shoulder_width=args.shoulder_width,
        shoulder_height=args.shoulder_height,
    )
