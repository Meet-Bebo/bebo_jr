"""Shared body-tracking utilities for image and camera pipelines."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import numpy as np

mp_pose = mp.solutions.pose


@dataclass(frozen=True)
class Point:
    """Named point stored as a homogeneous vector (x, y, z, 1)."""

    name: str
    vector: np.ndarray

    def transform(self, matrix: np.ndarray) -> "Point":
        """Return a new point transformed by the provided 4x4 matrix."""
        transformed = matrix @ self.vector
        return Point(name=self.name, vector=transformed)

    @property
    def xyz(self) -> np.ndarray:
        return self.vector[:3]


def points_from_landmarks(
    landmarks: landmark_pb2.NormalizedLandmarkList | None,
) -> list[Point]:
    """Convert MediaPipe landmarks into homogeneous points."""
    if landmarks is None:
        return []

    points: list[Point] = []
    for landmark_id, landmark in enumerate(landmarks.landmark):
        normalized = np.array([landmark.x, landmark.y, landmark.z, 1.0], dtype=float)
        point = Point(
            name=mp_pose.PoseLandmark(landmark_id).name,
            vector=normalized,
        )
        points.append(point)

    return points

def get_transformation_matrix(
    points: list["Point"], shoulder_width: float, shoulder_height: float
) -> np.ndarray:
    """
    Compute the transform that normalizes MediaPipe pose coordinates into a
    canonical 3D body space with:
        - canonical X : shoulder-to-shoulder direction
        - canonical Z : torso vertical direction (shoulder_mid -> hip_mid)
        - canonical Y : right-handed sideways direction
        - hips always below shoulders
        - correct scaling + translation
    """
    import numpy as np

    def normalize(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v * 0.0

    # --------------------------------------------------------------
    # 1. Extract key anatomical landmarks
    # --------------------------------------------------------------
    RS = np.array(next(p.xyz for p in points if p.name == "RIGHT_SHOULDER"), float)
    LS = np.array(next(p.xyz for p in points if p.name == "LEFT_SHOULDER"), float)
    RH = np.array(next(p.xyz for p in points if p.name == "RIGHT_HIP"), float)
    LH = np.array(next(p.xyz for p in points if p.name == "LEFT_HIP"), float)

    # --------------------------------------------------------------
    # 2. Build anatomic frame BEFORE scaling or translating
    # --------------------------------------------------------------
    shoulder_mid = 0.5 * (RS + LS)
    hip_mid = 0.5 * (RH + LH)

    # Canonical Z = torso direction (up/down)
    torso_vec = hip_mid - shoulder_mid
    ez_body = normalize(torso_vec)

    # Shoulder direction projected orthogonal to torso = canonical X
    shoulder_vec = RS - LS
    shoulder_vec_proj = shoulder_vec - np.dot(shoulder_vec, ez_body) * ez_body
    ex_body = normalize(shoulder_vec_proj)

    # Canonical Y = sideways (right-handed frame)
    ey_body = normalize(np.cross(ez_body, ex_body))

    # Re-orthonormalize ex
    ex_body = normalize(np.cross(ey_body, ez_body))

    # --------------------------------------------------------------
    # 3. Determine if torso is upside-down (hips above shoulders)
    # --------------------------------------------------------------
    R_body_to_world = np.column_stack((ex_body, ey_body, ez_body))
    R_world_to_body = R_body_to_world.T    # inverse of rotation

    RS_rot = R_world_to_body @ RS
    hip_rot = R_world_to_body @ hip_mid

    # correct orientation: hip_rot.z should be > RS_rot.z
    if hip_rot[2] > RS_rot[2]:
        # flip 180 deg around shoulder axis (canonical X)
        axis = ex_body
        R_flip = -np.eye(3) + 2 * np.outer(axis, axis)

        # apply flip to basis
        ex_body = R_flip @ ex_body
        ey_body = R_flip @ ey_body
        ez_body = R_flip @ ez_body

        # rebuild rotation matrices
        R_body_to_world = np.column_stack((ex_body, ey_body, ez_body))
        R_world_to_body = R_body_to_world.T

        # recompute rotated values for consistency
        RS_rot = R_world_to_body @ RS
        hip_rot = R_world_to_body @ hip_mid

    # --------------------------------------------------------------
    # 4. Compute scaling based on rotated shoulder width
    # --------------------------------------------------------------
    LS_rot = R_world_to_body @ LS
    current_width = abs(RS_rot[0] - LS_rot[0])
    scale = shoulder_width / current_width

    # --------------------------------------------------------------
    # 5. Translation to canonical height
    # --------------------------------------------------------------
    shoulder_mid_rot = R_world_to_body @ shoulder_mid

    translation = np.array([0, 0, shoulder_height]) - scale * shoulder_mid_rot

    # --------------------------------------------------------------
    # 6. Final transform matrix
    # --------------------------------------------------------------
    T = np.eye(4)
    T[:3, :3] = scale * R_world_to_body
    T[:3, 3] = translation

    # --------------------------------------------------------------
    # 7. Optional: rotate 180° around canonical Z axis
    # --------------------------------------------------------------
    Rz_180 = np.array([
        [-1,  0,  0],
        [ 0, -1,  0],
        [ 0,  0,  1],
    ], float)

    # rotate rotation
    T[:3, :3] = Rz_180 @ T[:3, :3]

    # rotate translation (in place)
    T[:3, 3] = Rz_180 @ T[:3, 3]

    return T

import numpy as np

def get_transformation_matrix_for_right_arm(
    arm_points: list["Point"], arm_length: float
) -> np.ndarray:
    """
    Create a transform that scales the entire arm such that:
        new_length = arm_length
    where the current arm length = |shoulder - elbow| + |elbow - wrist|.

    The shoulder is kept fixed (pivot point), so the transform is:
        T = T_shoulder * S * T_shoulder^{-1}
    """
    # --------------------------------------------------------------
    # 1. Extract needed points
    # --------------------------------------------------------------
    RS = np.array(next(p.xyz for p in arm_points if p.name == "RIGHT_SHOULDER"), float)
    RE = np.array(next(p.xyz for p in arm_points if p.name == "RIGHT_ELBOW"), float)
    RW = np.array(next(p.xyz for p in arm_points if p.name == "RIGHT_WRIST"), float)

    # --------------------------------------------------------------
    # 2. Compute current arm length
    # --------------------------------------------------------------
    seg1 = np.linalg.norm(RE - RS)
    seg2 = np.linalg.norm(RW - RE)
    current_len = seg1 + seg2

    if current_len < 1e-9:
        scale = 1.0
    else:
        scale = arm_length / current_len

    # --------------------------------------------------------------
    # 3. Build transform that scales around the shoulder pivot
    # --------------------------------------------------------------
    # Translation to origin: T1 (move shoulder to origin)
    T1 = np.eye(4)
    T1[:3, 3] = -RS

    # Scaling matrix
    S = np.eye(4)
    S[0,0] = S[1,1] = S[2,2] = scale

    # Translate back: T2 (restore shoulder position)
    T2 = np.eye(4)
    T2[:3, 3] = RS

    # Final transform
    T = T2 @ S @ T1
    return T

def get_transformation_matrix_for_left_arm(
    arm_points: list["Point"], arm_length: float
) -> np.ndarray:
    """
    Create a transform that scales the entire arm such that:
        new_length = arm_length
    where the current arm length = |shoulder - elbow| + |elbow - wrist|.

    The shoulder is kept fixed (pivot point), so the transform is:
        T = T_shoulder * S * T_shoulder^{-1}
    """
    # --------------------------------------------------------------
    # 1. Extract needed points
    # --------------------------------------------------------------
    LS = np.array(next(p.xyz for p in arm_points if p.name == "LEFT_SHOULDER"), float)
    LE = np.array(next(p.xyz for p in arm_points if p.name == "LEFT_ELBOW"), float)
    LW = np.array(next(p.xyz for p in arm_points if p.name == "LEFT_WRIST"), float)

    # --------------------------------------------------------------
    # 2. Compute current arm length
    # --------------------------------------------------------------
    seg1 = np.linalg.norm(LE - LS)
    seg2 = np.linalg.norm(LW - LE)
    current_len = seg1 + seg2

    if current_len < 1e-9:
        scale = 1.0
    else:
        scale = arm_length / current_len

    # --------------------------------------------------------------
    # 3. Build transform that scales around the shoulder pivot
    # --------------------------------------------------------------
    # Translation to origin: T1 (move shoulder to origin)
    T1 = np.eye(4)
    T1[:3, 3] = -LS

    # Scaling matrix
    S = np.eye(4)
    S[0,0] = S[1,1] = S[2,2] = scale

    # Translate back: T2 (restore shoulder position)
    T2 = np.eye(4)
    T2[:3, 3] = LS

    # Final transform
    T = T2 @ S @ T1
    return T

DEFAULT_VISUALIZATION_POINTS: tuple[str, ...] = (
    "RIGHT_SHOULDER",
    "RIGHT_ELBOW",
    "RIGHT_WRIST",
    "LEFT_SHOULDER",
    "LEFT_ELBOW",
    "LEFT_WRIST",
)


def select_named_points(
    points: Iterable[Point], required_names: Iterable[str] = DEFAULT_VISUALIZATION_POINTS
) -> dict[str, Point] | None:
    """Return a mapping for the requested point names if all exist."""
    required = tuple(required_names)
    mapping: dict[str, Point] = {}

    for point in points:
        if point.name in required:
            mapping[point.name] = point

    if len(mapping) != len(required):
        return None

    return mapping
