"""Track a live camera pose and stream it to the Placo visualizer + robot view."""

from __future__ import annotations

import argparse
import cv2
import mediapipe as mp
import placo

from placo_utils.visualization import frame_viz, point_viz, points_viz, robot_viz

from body_tracking_common import (
    DEFAULT_VISUALIZATION_POINTS,
    Point,
    get_transformation_matrix_for_left_arm,
    get_transformation_matrix_for_right_arm,
    get_transformation_matrix,
    points_from_landmarks,
    select_named_points,
)

DEFAULT_SHOULDER_WIDTH = 0.178  # meters (derived from robot URDF)
DEFAULT_SHOULDER_HEIGHT = 0.179517  # meters (base -> shoulder frame z-offset)
DEFAULT_ARM_LENGTH = 0.12  # meters (approximate)


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
        default=DEFAULT_SHOULDER_WIDTH,
        help="Desired shoulder width of the normalized skeleton (meters).",
    )
    parser.add_argument(
        "--shoulder-height",
        type=float,
        default=DEFAULT_SHOULDER_HEIGHT,
        help="Height (meters) to place the normalized shoulder midpoint.",
    )
    parser.add_argument(
        "--urdf-path",
        type=str,
        default="../simulation/robot.urdf",
        help="Path to the robot URDF for visualization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {args.camera_index}")

    robot = placo.RobotWrapper(args.urdf_path, placo.Flags.ignore_collisions)
    viz = robot_viz(robot)
    viz.display(robot.state.q)

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
                if points:
                    transform = get_transformation_matrix(
                        points, args.shoulder_width, args.shoulder_height
                    )

                    transformed_points = [point.transform(transform) for point in points]

                    right_arm_transform = get_transformation_matrix_for_right_arm(
                        transformed_points, DEFAULT_ARM_LENGTH
                    )
                    left_arm_transform = get_transformation_matrix_for_left_arm(
                        transformed_points, DEFAULT_ARM_LENGTH
                    )


                    # for all points in transformed_points, if point is in right arm, apply right_arm_transform
                    # Only apply to shoulder, elbow, wrist
                    for i, point in enumerate(transformed_points):
                        if point.name.startswith("RIGHT_") and point.name in ["RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"]:
                            transformed_points[i] = point.transform(
                                right_arm_transform
                            )
                        elif point.name.startswith("LEFT_") and point.name in ["LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"]:
                            transformed_points[i] = point.transform(
                                left_arm_transform
                            )

                    named_points = select_named_points(
                        transformed_points, DEFAULT_VISUALIZATION_POINTS
                    )
                    if named_points:
                        publish_points(named_points)

                viz.display(robot.state.q)

        except KeyboardInterrupt:
            print("Stopping overlay.")
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

    points_viz("right_point", right_points, radius=0.01, color=0x0000FF)
    points_viz("left_point", left_points, radius=0.01, color=0xFF0000)

if __name__ == "__main__":
    main()
