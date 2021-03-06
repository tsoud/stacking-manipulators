import json
from dataclasses import asdict, dataclass, field
from typing import Callable, Tuple, Optional

import numpy as np

from robotic_stacking.bullet_envs import single_agent_stacking

# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class single_env_config:
    """
    Customizable configuration for a single-agent stacking env.

    NOTE: Attributes cannot be changed once the instance is created. 
    To change the configuration, overwrite the instance.
    """
    robot_pose_initialization: str
    random_position_limits: Tuple
    random_orientation_limits: Tuple
    num_cubes: int
    num_targets: int
    target_formation: str
    use_GUI: bool
    episode_time_limit: int
    reset_on_episode_end: bool
    track_pose_error: bool
    track_per_step_cube_mvmt: bool
    apply_collision_penalties: bool
    n_transition_steps_per_sec: int
    robot_base_position: Optional[Tuple] = None
    robot_base_orientation: Optional[Tuple] = None
    robot_kwargs: Optional[dict] = None
    starting_cube_locations : Optional[np.array] = None
    starting_cube_orientations : Optional[np.array] = None
    cube_pose_params: Optional[dict] = None
    target_formation_coords: Optional[np.ndarray] = None
    target_formation_position: Optional[Tuple] = None
    target_cube_orientation: Optional[Tuple] = None
    gravity: Tuple = (0., 0., -9.81)
    simulation_steps_per_sec: int = 240
    mask_actions: Optional[Tuple] = None
    reward_function: str = 'dense'

    def to_env_kwargs(self) -> dict:
        return asdict(self)

    def save_to_json(self, file_path:str):
        with open(file_path, 'w') as f:
            json.dump(self.to_env_kwargs(), fp=f)

    def to_env(self):
        """Returns the configured environment."""
        pass

@dataclass(frozen=True)
class default_kvG3_stacking_single_env(single_env_config):
    """Default Kinova Gen3 Hand-E stacking configuration."""
    robot_pose_initialization: str = 'random'
    random_position_limits: Tuple = (-0.1, 0.1)
    random_orientation_limits: Tuple = (-0.25*np.pi, 0.25*np.pi)
    num_cubes: int = 8
    num_targets: int = 8
    target_formation: str ='default_4_corners'
    use_GUI: bool = False
    episode_time_limit: int = 120
    reset_on_episode_end: bool = False
    track_pose_error: bool = False
    track_per_step_cube_mvmt: bool = False
    apply_collision_penalties: bool = False
    n_transition_steps_per_sec: int = 10

    def to_env(self):
        """Returns the configured environment."""
        return (
            single_agent_stacking.single_kvG3_7DH_stacking_env(
                **self.to_env_kwargs()
            )
        )


@dataclass(frozen=True)
class kvG3_stacking_5action(default_kvG3_stacking_single_env):
    """
    Config with EE actions limited to dx, dy, dz, Rz, and d_grip.
    """
    mask_actions: Tuple = field(default=(1, 1, 1, 0, 0, 1, 1))


@dataclass(frozen=True)
class kvG3_stacking_pyramid(default_kvG3_stacking_single_env):
    """
    Config with the default pyramid target structure.
    """
    num_cubes: int = 15
    num_targets: int = 14
    target_formation: str ='default_pyramid'


@dataclass(frozen=True)
class kvG3_stacking_pyramid_5a(kvG3_stacking_5action):
    """
    Pyramid config with dx, dy, dz, Rz, and d_grip actions only.
    """
    num_cubes: int = 15
    num_targets: int = 14
    target_formation: str = 'default_pyramid'

