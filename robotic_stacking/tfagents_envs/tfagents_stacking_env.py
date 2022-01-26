import json
from typing import List, Optional, Union

import numpy as np
# import tensorflow as tf
import tf_agents.environments.utils as tfa_env_utils
from PIL import Image
from tf_agents import specs
from tf_agents.environments import py_environment, tf_py_environment
from tf_agents.trajectories import time_step

from robotic_stacking.bullet_envs import env_configs
from robotic_stacking import utils

# --------------------------------------------------------------------------- #


class tfagents_stacking_env(py_environment.PyEnvironment):
    """
    TF-Agents wrapper for the robotic stacking environment.

    keyword args:
    ------------
    action_bounds: lower and upper bounds [-/+ dx, -/+ dy, -/+ dz, 
        -/+ Rx, -/+ Ry, -/+ Rz, -/+ d_grip] for agnet actions. If the 
        env restricts (i.e. masks) some actions, only applicable 
        actions are used.
    stacking_env_config: a stacking environment configuration with a 
        `to_env()` method.
    config_kwargs: optional dict of keyword args to pass to the 
        config constructor.
    """
    def __init__(self, 
                 stacking_env_config:env_configs.single_env_config, 
                 action_bounds_lower:Union[List, np.array], 
                 action_bounds_upper:Union[List, np.array], 
                 config_kwargs:Optional[dict]=None):

        super().__init__()
        config_kwargs = {} if config_kwargs is None else config_kwargs
        self._env_config = stacking_env_config(**config_kwargs)
        self._env = self._env_config.to_env()
        # determine action space specs from configuration
        self._available_actions = self._env._available_actions
        self._action_space_shape = self._available_actions[0].shape
        if len(action_bounds_lower) != len(action_bounds_upper):
            raise ValueError(
                'Length of list or array for upper and lower action '
                + 'bounds are not equal.'
            )
        if len(action_bounds_lower) != self._action_space_shape[0]:
            raise utils.IncorrectNumberOfArgs(
                'Length of action bounds should match the number of available'
                + ' actions in the configuration. '
                + f'There are {self._action_space_shape[0]} available actions'
                + f' but {len(action_bounds_lower)} bound values were given.'
            )
        self._action_bounds_lower = action_bounds_lower
        self._action_bounds_upper = action_bounds_upper
        # define action spec
        self._action_spec = specs.array_spec.BoundedArraySpec(
            shape=self._action_space_shape,
            dtype=np.float32,
            minimum=self._action_bounds_lower,
            maximum=self._action_bounds_upper,
            name='action'
        )
        # launch the environment
        self._env.make()
        # get the initial state
        _, self._episode_done, self._observations = self.get_state_data()
        # define the observation spec
        self._observation_spec = {
            'episode_done': specs.array_spec.ArraySpec(
                shape=self._episode_done.shape,
                dtype=self._episode_done.dtype
            ),
            'observations': specs.array_spec.ArraySpec(
                shape=self._observations.shape,
                dtype=self._observations.dtype
            )
        }

    @classmethod
    def create_from_json(cls, json_filepath):
        with open(json_filepath, 'r') as fp:
            config_dict = json.load(fp=fp)
        return cls(**config_dict)

    def get_state_data(self):
        """
        Extract observations, reward, episode status from the state.
        """
        # transition reward
        reward = np.array(self._env._episode_reward, dtype=np.float32)
        # episode end
        end = np.array(self._env._episode_done, dtype=np.bool_)
        # state observations
        observations = np.concatenate(
            (
                # end-effector observations
                self._env._current_ee_pos,          # ee position
                self._env._current_ee_ort,          # ee orientation
                self._env._step_pose_error,         # desired vs actual pose
                self._env._current_grip_pos,        # gripper finger position
                self._env._grip_force,              # grip force
                # flattened cube and target observations
                self._env._cube_point_locations.reshape(-1), 
                self._env._target_point_locations.reshape(-1), 
                # collisions
                (
                    self._env._collision_intensity_self,
                    self._env._collision_intensity_floor,
                ), 
                # cubes contacting grip fingers
                self._env._grip_touching_cube.reshape(-1),
                # cubes aligned with targets
                self._env._cubes_aligned_w_targets
            )
        )
        # cast all observations 32-bit floats per tf defaults
        return reward, end, observations.astype(np.float32)

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._env.reset()
        # retrieve initial state
        _, self._episode_done, self._observations = self.get_state_data()

        return time_step.restart(
            {
                'episode_done': self._episode_done, 
                'observations': self._observations
            }
        )

    def _step(self, action):
        if self._episode_done:    
            return self.reset()

        else:
            self._env.run_actions(action)
            reward, self._episode_done, self._state = self.get_state_data()
            new_state = {
                'episode_done': self._episode_done,
                'observations': self._observations
            }
            if self._episode_done:
                return time_step.termination(new_state, reward)

            return time_step.transition(new_state, reward)

    def render(self, 
               show:bool=True, 
               save_to:Optional[str]=None, 
               img_size:tuple=(960, 512), 
               camera_view:tuple=([-0.1, 0.3, 0.15], 1.5, 23., -15., 0., 2), 
               projection_matrix:Optional[tuple]=None):
        """
        Render a frame. Used mainly for debugging.

        keyword args:
        ------------
        show: show the image in a viewer (depends on system).
        save_to: if a filepath is given, a .png is saved to the file.
        img_size: (image width, image height) in pixels.
        camera_view: Parameters to specify camera location and view 
            if desired. It is often best to keep defaults.
            Params:
            [camera focal point x, y, z], distance to focus, 
            camera yaw, camera pitch, camera roll, vertical axis 
            (1 == Y, 2 == Z)
        projection_matrix: An optional 4x4 projection matrix flattened 
            to a 16-element tuple. Unless the user is very familiar 
            with OpenGL rendering, it is strongly recommended to keep 
            the default values.
        """
        view_matrix = self._env.sim.computeViewMatrixFromYawPitchRoll(*camera_view)      
        if projection_matrix is None:
            proj_matrix = self._env.sim.computeProjectionMatrixFOV(
                60, (img_size[0]/img_size[1]), 0.1, 100
            )
        else:
            proj_matrix = projection_matrix
        img_data = self._env.sim.getCameraImage(
            img_size[0], img_size[1],
            view_matrix, proj_matrix,
            shadow=0, flags=4
        )
        img = Image.fromarray(img_data[2], "RGBA")
        if save_to:
            img.save(save_to)
        if show:
            img.show()

        return img

    def close(self):
        self._env.close()

    def wrap_to_TF_env(self, 
                       validation:Optional[int]=5, 
                       check_dims:bool=True, 
                       isolation:bool=False):
        """
        Wrap a the PyEnvironment `env` into in-graph TF environment.

        keyword args:
        ------------
        validation: Validate `env` by running `n` episodes before wrapping. 
                    If `None`, no validation is performed.
        check_dims, isolation: Parameters passed to `TFPyEnvironment`.
        """
        if validation is not None:
            tfa_env_utils.validate_py_environment(self, episodes=validation)
        tf_env = tf_py_environment.TFPyEnvironment(
            self, check_dims=check_dims, isolation=isolation
        )

        return tf_env

