import json
import pprint
from collections import namedtuple
from copy import deepcopy
from typing import List, Optional, Union

import numpy as np
import tf_agents.environments.utils as tfa_env_utils
from PIL import Image
from tf_agents import specs
from tf_agents.environments import py_environment, tf_py_environment
from tf_agents.trajectories import time_step

from robotic_stacking.bullet_envs import env_configs, single_agent_stacking
from robotic_stacking import utils

# --------------------------------------------------------------------------- #


class tfagents_stacking_env(py_environment.PyEnvironment):
    """
    TF-Agents wrapper for the robotic stacking environment.

    keyword args:
    ------------
    stacking_env_config: a stacking environment configuration with a 
        `to_env()` method.
    action_bounds: lower and upper bounds [-/+ dx, -/+ dy, -/+ dz, 
        -/+ Rx, -/+ Ry, -/+ Rz, -/+ d_grip] for agnet actions. If the 
        env restricts (i.e. masks) some actions, only applicable 
        actions are used.
    reset_env_to_random: If `True`, the robot is initialized with a 
        random new pose whenever `reset()` is called. Only applied 
        if `robot_pose_initialization` is set to 'random' in the env 
        configuration.
    config_kwargs: optional dict of keyword args to pass to the 
        config constructor.
    """
    def __init__(self, 
                 stacking_env:single_agent_stacking.single_kvG3_7DH_stacking_env,
                 action_bounds_lower:Union[List, np.array],
                 action_bounds_upper:Union[List, np.array],
                 reset_env_to_random:bool=False):

        super().__init__()

        self._env = stacking_env
        if not hasattr(self._env, '_sim_id'):
            self._env.make()

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
                + ' actions in the configuration.'
                + f' There are {self._action_space_shape[0]} available actions'
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
        # get the initial state
        _, self._episode_done, self._observations = self.get_state_data()
        # define the observation spec
        self._observation_spec = specs.array_spec.ArraySpec(
            shape=self._observations.shape,
            dtype=self._observations.dtype
        )
        # env info to pass to __repr__()
        self.__env_info = namedtuple(
            'env_config', [
                'robot', 'n_cubes', 'target_formation', 'transitions_per_sec', 
                'sim_steps_per_sec', 'max_transitions_per_episode', 
                'available_actions'
                ]
        )
        # arg for `reset()` method
        self.reset_to_random = reset_env_to_random

    @classmethod
    def from_env_configuration(cls, 
                               env_configuration:env_configs.single_env_config,
                               action_bounds_lower:Union[List, np.array],
                               action_bounds_upper:Union[List, np.array],
                               reset_env_to_random:bool=False,
                               config_kwargs:Optional[dict]=None):
        """
        Create the TF environment directly from a configuration.

        keyword args:
        ------------
        env_config: a stacking environment configuration with a 
            `to_env()` method.
        action_bounds: lower and upper bounds [-/+ dx, -/+ dy, -/+ dz, 
            -/+ Rx, -/+ Ry, -/+ Rz, -/+ d_grip] for agnet actions. If the 
            env restricts (i.e. masks) some actions, only applicable 
            actions are used.
        reset_env_to_random: If `True`, the robot is initialized with a 
            random new pose whenever `reset()` is called. Only applied 
            if `robot_pose_initialization` is set to 'random' in the env 
            configuration.
        config_kwargs: optional dict of keyword args to pass to the 
            config constructor.
        """
        env_config = env_configuration
        config_kwargs = {} if config_kwargs is None else config_kwargs
        new_env = env_config(**self.config_kwargs).to_env()
        return cls(
            stacking_env=new_env, 
            action_bounds_lower=action_bounds_lower, 
            action_bounds_upper=action_bounds_upper,
            reset_env_to_random=reset_env_to_random
        )

    def copy_env(self):
        """Returns a facsimile of the tf_env."""
        env_copy = self._env.copy_env()

        return tfagents_stacking_env(
            stacking_env=env_copy, 
            action_bounds_lower=self._action_bounds_lower, 
            action_bounds_upper=self._action_bounds_upper, 
            reset_env_to_random=self.reset_to_random,
        )

    def __repr__(self):
        actions = np.array(
            ['dx', 'dy', 'dz', 'Rx', 'Ry', 'Rz', 'd_grip']
        )[self._env._available_actions]
        env_config_info = self.__env_info(
            self._env.arm_control, 
            self._env.num_cubes, 
            self._env.target_formation, 
            self._env.transition_steps_per_sec, 
            self._env.sim_steps_per_sec, 
            int(self._env._episode_sim_step_limit/self._env._n_substeps), 
            tuple(actions)
        )
        return pprint.pformat(env_config_info._asdict(), width=80, sort_dicts=False)

    @property
    def env(self):
        return self._env
    
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

    @property
    def env(self):
        return self._env
    
    def _reset(self):
        self._env.reset(self.reset_to_random)
        # retrieve initial state
        _, self._episode_done, self._obs = self.get_state_data()

        return time_step.restart(self._obs)

    def _step(self, action):
        if self._episode_done:    
            return self.reset()

        else:
            self._env.run_actions(action)
            reward, self._episode_done, self._obs = self.get_state_data()
            if self._episode_done:
                return time_step.termination(self._obs, reward)

            return time_step.transition(self._obs, reward)

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
                       validation:Optional[int]=None, 
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

