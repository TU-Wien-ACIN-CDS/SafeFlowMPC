from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..RobotModel import RobotModel
from ..utils import compute_polytope_vertices, make_box, normalize_set_size


@dataclass
class ObstacleDescription:
    set: List[List[np.ndarray]]
    points: np.ndarray
    bbox: np.ndarray


class ObstacleManager:
    """Manages obstacle definitions and collision checking."""

    def __init__(self):
        self.obstacles = {}

    def add_default_obstacles(self) -> None:
        """Add default obstacle configuration."""

        obstacles = [
            [0.45, -0.06, 0.725, 1.1, 0.06, 0.9],
            [0.45, -0.06, 0.0, 1.1, 0.06, 0.38],
            # [-0.5, -1.0, 0.0, -0.25, 1.0, 1.5],
            # [-1.0, -1.0, 1.2, 1.0, 1.0, 1.5],
            # [-1.0, -1.0, -0.5, 1.0, 1.0, -0.01],
        ]
        for i, ob in enumerate(obstacles):
            set_ob = make_box(ob[:3], ob[3:])
            points = np.array(compute_polytope_vertices(set_ob[0], set_ob[1]))
            bbox = np.concatenate((np.min(points, axis=0), np.max(points, axis=0)))

            self.obstacles[f"obs{i}"] = ObstacleDescription(
                set=normalize_set_size([set_ob])[0], points=points, bbox=bbox
            )

    def get_obstacles(self) -> List[List[float]]:
        """Return current obstacle configuration."""
        obs_sets = [x.set for x in self.obstacles.values()]
        obs_points_sets = [x.points for x in self.obstacles.values()]
        bboxs = [x.bbox for x in self.obstacles.values()]
        return obs_sets, obs_points_sets, bboxs

    def add_oriented_box(self, p_goal, box_dimensions, id=0):
        """
        Create an oriented box based on the rotation matrix of the goal pose.

        Args:
            p_goal (tuple): Goal pose (position and rotation matrix).
            box_dimensions (tuple): Dimensions of the box (length, width, height).

        Returns:
            tuple: (A, b) where Ax <= b defines the oriented box.
        """
        # Extract the rotation matrix from the goal pose
        position = p_goal[:3]
        rotation_matrix = R.from_rotvec(p_goal[3:]).as_matrix()

        # Construct the oriented box constraints
        a_set = np.empty((6, 3))
        b_set = np.empty(6)

        # Construct the constraints for the oriented box
        a_set[:3, :] = np.eye(3)
        a_set[3:, :] = -np.eye(3)
        b_set[:3] = box_dimensions / 2
        b_set[3:] = box_dimensions / 2
        a_set = a_set @ rotation_matrix.T
        b_set = b_set + a_set @ position

        set_ob = normalize_set_size([[a_set, b_set]])[0]
        points = np.array(compute_polytope_vertices(a_set, b_set))
        bbox = np.concatenate((np.min(points, axis=0), np.max(points, axis=0)))

        self.obstacles[f"oriented_obs{id}"] = ObstacleDescription(
            set=normalize_set_size([set_ob])[0], points=points, bbox=bbox
        )
