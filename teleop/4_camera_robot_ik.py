"""Track camera pose data and drive the robot wrists with IK tasks."""

from __future__ import annotations

import argparse
import cv2
import mediapipe as mp
import placo
import numpy as np

from placo_utils.tf import tf
from placo_utils.visualization import frame_viz, point_viz, robot_frame_viz, robot_viz

from body_tracking_common import (
    DEFAULT_VISUALIZATION_POINTS,
    Point,
    get_transformation_matrix_for_left_arm,
    get_transformation_matrix_for_right_arm,
    get_transformation_matrix,
    points_from_landmarks,
    select_named_points,
)

DEFAULT_SHOULDER_WIDTH = 0.18  # meters
DEFAULT_SHOULDER_HEIGHT = 0.18  # meters
DEFAULT_ARM_LENGTH = 0.12  # meters
DEFAULT_DT = 0.01

def main() -> None:

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index 0")

    robot = placo.RobotWrapper("../simulation/robot.urdf", placo.Flags.ignore_collisions)
    viz = robot_viz(robot)
    viz.display(robot.state.q)

    solver = placo.KinematicsSolver(robot)
    solver.mask_fbase(True)
    solver.dt = 0.001

    # Effector tasks for wrists
    effector_task_left = solver.add_position_task("left_hand", np.array([0., 0., 0.]))
    effector_task_left.configure("left_hand", "soft", 1.0)

    effector_task_right = solver.add_position_task("right_hand", np.array([0., 0., 0.]))
    effector_task_right.configure("right_hand", "soft", 1.0)

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

                transform = get_transformation_matrix(
                    points, DEFAULT_SHOULDER_WIDTH, DEFAULT_SHOULDER_HEIGHT
                )
                transformed_points = [point.transform(transform) for point in points]

                right_arm_transform = get_transformation_matrix_for_right_arm(
                    transformed_points, DEFAULT_ARM_LENGTH
                )
                left_arm_transform = get_transformation_matrix_for_left_arm(
                    transformed_points, DEFAULT_ARM_LENGTH
                )

                for idx, point in enumerate(transformed_points):
                    if point.name in {"RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"}:
                        transformed_points[idx] = point.transform(right_arm_transform)
                    elif point.name in {"LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"}:
                        transformed_points[idx] = point.transform(left_arm_transform)

                named_points = select_named_points(
                    transformed_points, DEFAULT_VISUALIZATION_POINTS
                )
                if not named_points:
                    continue

                update_effector_tasks(named_points, effector_task_left, effector_task_right)
                publish_points(named_points)

                solver.solve(True)
                robot.update_kinematics()
                viz.display(robot.state.q)
        except KeyboardInterrupt:
            print("Stopping IK teleoperation.")
        finally:
            cap.release()


def update_effector_tasks(
    points: dict[str, Point],
    effector_task_left,
    effector_task_right,
) -> None:
    """Update IK targets for both wrists."""
    T_left = tf.translation_matrix(points["LEFT_WRIST"].xyz)
    T_right = tf.translation_matrix(points["RIGHT_WRIST"].xyz)

    effector_task_left.target_world = points["LEFT_WRIST"].xyz
    effector_task_right.target_world = points["RIGHT_WRIST"].xyz

    frame_viz("target_left", T_left)
    frame_viz("target_right", T_right)


def publish_points(points: dict[str, Point]) -> None:
    """Push wrist positions to the Placo point visualizer."""
    point_viz("left_wrist_point", points["LEFT_WRIST"].xyz, radius=0.015, color=0xFF0000)
    point_viz(
        "right_wrist_point",
        points["RIGHT_WRIST"].xyz,
        radius=0.015,
        color=0x0000FF,
    )


if __name__ == "__main__":
    main()
