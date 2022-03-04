import warnings
from typing import Callable, Iterable, Literal, Optional, Tuple

import numpy as np
import quaternionic as qtr

from robotic_stacking import robot, utils
from robotic_stacking.bullet_envs import env_utils
from robotic_stacking.bullet_envs.env_objects import small_cube, virtual_cube
from robotic_stacking.bullet_envs.base_simulations import single_agent_env

# ----------------------------------------------------------------------------
# A single robot arm tasked with stacking scattered cubes to align with a
# target formation.
# ----------------------------------------------------------------------------


class single_kvG3_7DH_stacking_env(single_agent_env):
    """
    Single agent stacking environment with a Kinova Gen3 HandE robot.

    The robot arm has to pick from cubes scattered randomly around its 
    workspace and stack them in the target formation.

    The robot has 7 available actions:
    Move the end-effector a given amount in the x, y, and/or z 
    direction (3 actions, +/-, continuous).
    Rotate the end-effector by a given amount about the 
    x, y, and/or z axis, corresponding to pitch, yaw and roll 
    respectively (3 actions, +/-, continuous).
    Open or close the gripper by a given amount (1 action, +/-, 
    continuous).

    The action space is: [dx, dy, dz, dRx, dRy, dRz, d_grips]. 
    The action space can be restricted with a boolean mask. 
    For example, [1, 1, 1, 0, 0, 0, 1] restricts actions to only 
    translational end-effector movements and opening or closing 
    the gripper.

    By default, the robot uses its local coordinate system to define 
    translations. To change this, pass `{use_robot_lcs: False}` 
    to `robot_kwargs`. 

    keyword args:
    ------------
    robot_pose_initialization: a method of initializing the robot 
        base position and orientation.
    random_position_limits: range limits for the position when using 
        'random' initialization.
    random_orientation_limits: range limits for the orientation (about 
        the z-axis) when using 'random' initialization.
    robot_base_position: base position when 'specified' initialization 
        is chosen. Defaults to origin if 'specified' is chosen and no 
        position is given.
    robot_base_orientation: base position when 'specified' initialization 
        is chosen. Defaults to zero rad if 'specified' is chosen and no 
        position is given.
    robot_kwargs: keyword args to the robot controller.
    num_cubes: number of physical cubes to stack.
    starting_cube_locations:
    starting_cube_orientations:
    cube_pose_params: Parameters to pass to the `set_cube_poses()` 
        method for randomly initializing cube locations. If `None`, 
        a default group of parameters is selected.
    target_formation: a target stack formation. A couple of preset 
        default options are available. 'default_4_corners' gives 
        4 vertical stacks spaced evenly from each other.
        'default_pyramid' is a 3-2-1 stack.
    target_formation_coords: direct coordinates of target locations 
        when using 'specified' option.
    target_formation_position: select a position for the target stack 
        "centroid". Defaults to (0., 0., 0.) for 'default_4_corners' 
        and (0.5, 0., 0.) for 'default_pyramid' if none is given. 
    target_cube_orientation: Select an orientation for the stacked 
        cubes. Default is zero rad about all axes.
    use_GUI: run the simulation in a graphical window.
    gravity: x, y, z gravity values, z is normal to ground.
    n_transition_steps_per_sec: the number of transition steps made by 
        an agent in a second. This is combined with 
        `simulation_steps_per_sec` to calculate the number of substeps 
        in a transition step.
    simulation_steps_per_sec: the PyBullet simulation timestep.
    episode_time_limit: time (in seconds) to allow before ending an 
        episode and resetting.
    reset_on_episode_end: reset the environment if episode times out 
        or all targets are achieved. It is best to leave this `False` 
        when using the environment in RL training to let the RL 
        library handle resetting.
    mask_actions: Use a boolean mask to restrict available actions. 
    track_pose_error: keep track of error between desired and actual 
        end-effector pose.
    track_per_step_cube_mvmt: track Euclidean distance moved by cubes 
        in transition steps. This can be used for reward shaping. 
        Defaults to `True` if `reward_function` is 'dense'.
    apply_collision_penalties: apply a penalty for self-collisions or 
        collisions between arm and floor. Defaults to `True` if 
        `reward_function` is 'dense'.
    reward_function: method for calculating rewards. 'sparse' uses a 
        binary reward that is negative for every step and positive 
        when the task is completed. Sparse rewards are scaled to +/- 1
        'dense' applies additional intermediate penalties and rewards.
        Rewards are not scaled when 'dense' is used. 'custom' allows 
        the user to pass a custom reward fn to the `custom_rewards()` 
        method that will be used for reward calculation.
    """
    def __init__(self, 
                 robot_pose_initialization:
                    Literal['random', 'specified', 'origin']='random',
                 random_position_limits:Tuple=(-0.1, 0.1),
                 random_orientation_limits:Tuple=(-0.25*np.pi, 0.25*np.pi),
                 robot_base_position:Optional[Tuple]=None,
                 robot_base_orientation:Optional[Tuple]=None,
                 robot_kwargs:Optional[dict]=None,
                 num_cubes:int=8,
                 starting_cube_locations:Optional[np.array]=None,
                 starting_cube_orientations:Optional[np.array]=None,
                 cube_pose_params:Optional[dict]=None,
                 num_targets:int=8,
                 target_formation:
                    Literal['default_4_corners', 'default_pyramid', 'specified']='default_4_corners',
                 target_formation_coords:Optional[np.ndarray]=None,
                 target_formation_position:Optional[Tuple]=None,
                 target_cube_orientation:Optional[Tuple]=None,
                 use_GUI:bool=False,
                 gravity:Tuple=(0., 0., -9.8),
                 n_transition_steps_per_sec:int=10,
                 simulation_steps_per_sec:int=240,
                 episode_time_limit:int=120,
                 reset_on_episode_end:bool=False,
                 mask_actions:Optional[Tuple]=None,
                 track_pose_error:bool=False,
                 track_per_step_cube_mvmt:bool=False,
                 apply_collision_penalties:bool=False,
                 reward_function:
                    Literal['sparse', 'dense', 'custom']='dense'):
        # maintain initial arg values for copying
        self.__initial_arg_vals = locals().copy()
        self.__initial_arg_vals = {
            k: v for k, v in self.__initial_arg_vals.items() if not k == 'self'
        }
        # set up robot pose
        self._robot_pose_init=robot_pose_initialization
        if self._robot_pose_init == 'random':
            self.rand_pos_lims = random_position_limits    
            self.rand_ort_lims = random_orientation_limits
            self.robot_base_position = (
                np.random.uniform(*self.rand_pos_lims), 
                np.random.uniform(*self.rand_pos_lims), 
                0.
            )
            self.robot_base_orientation = utils.quaternion_from_RxRyRz(
                0., 0., np.random.uniform(*self.rand_ort_lims)
            )
        elif self._robot_pose_init == 'specified':
            self.robot_base_position = (
                robot_base_position if robot_base_position else (0., 0., 0.)
            )
            self.robot_base_orientation = (
                robot_base_orientation if robot_base_orientation 
                else (0., 0., 0., 1.)
            )
        elif self._robot_pose_init == 'origin':
            self.robot_base_position = (0., 0., 0.)
            self.robot_base_orientation = (0., 0., 0., 1.)
        else:
            raise ValueError(
                f"`robot_pose_initialization={robot_pose_initialization}` is "
                + "invalid. Acceptable args are 'random', 'specified' and"
                + " 'origin'."
            )
        self.robot_kwargs = robot_kwargs
        # set additional simulation parameters
        self.num_cubes = num_cubes
        self.starting_cube_locations = starting_cube_locations
        self.starting_cube_orientations = starting_cube_orientations
        self.cube_pose_params = {
                    'min_dist': 0.2, 'max_dist': 0.7, 
                    'sweep': (1.5*np.pi), 
                    'delta_z_rotations': [-0.5*np.pi, 0.5*np.pi], 
                    'center': self.robot_base_position[:2]
                }
        if cube_pose_params is not None:
            for k in cube_pose_params.keys():
                if k not in self.cube_pose_params.keys():
                    raise KeyError(
                        f"Parameter '{k}' in `cube_pose_params` is invalid."
                    )
            self.cube_pose_params.update(cube_pose_params)
        self.num_targets = num_targets
        self.target_formation = target_formation                      
        if self.target_formation == 'specified':
            self.target_coords = target_formation_coords
            if self.target_coords is None:
                self.target_formation = 'default_4_corners'
        elif self.target_formation == 'default_4_corners':
            if target_formation_position is None:
                self.target_ctrd_pos = (0., 0., 0.)
            else:
                self.target_ctrd_pos = target_formation_position
            self.target_coords = env_utils.four_corner_structure(
                self.target_ctrd_pos, self.num_cubes, cube_spacing=8
            )
        elif self.target_formation == 'default_pyramid':
            if target_formation_position is None:
                self.target_ctrd_pos = (0.5, 0., 0.)
            else:
                self.target_ctrd_pos = target_formation_position
            self.target_coords = env_utils.simple_pyramid_structure(
                self.target_ctrd_pos, level_sizes=[3, 2, 1]
            )
            # adjust if pyramid size exceeds num_targets
            max_idx = min(self.num_targets, len(self.target_coords))
            self.target_coords = self.target_coords[:max_idx]
        else:
            raise ValueError(
                f'`target_formation={target_formation}` is invalid. '
                + 'Acceptable args are \'default_4_corners\', '
                + '\'default_pyramid\' and \'specified\'.'
            )
        self.target_cube_ort = target_cube_orientation
        # initialize time step parameters
        self.sim_steps_per_sec = simulation_steps_per_sec
        self.transition_steps_per_sec = n_transition_steps_per_sec
        self._n_substeps = env_utils.calculate_simulation_substeps(
            self.transition_steps_per_sec, self.sim_steps_per_sec
        )
        self._episode_sim_step_limit = episode_time_limit*simulation_steps_per_sec
        self._max_transition_steps = episode_time_limit*n_transition_steps_per_sec
        self._reset_on_episode_end = reset_on_episode_end
        # environment parameters
        self._set_GUI = use_GUI
        self._set_gravity = gravity
        # initialize action space params
        if mask_actions is None: 
            mask_actions = np.array([True]*7, dtype=bool)
        else:
            if len(mask_actions) != 7:
                raise utils.IncorrectNumberOfArgs(
                    '`mask_actions` requires an iterable with 7 elements.'
                )
            mask_actions = np.array(mask_actions, dtype=bool)
        self._available_actions = mask_actions.nonzero()
        # initialize error tracking
        self.track_pose_error = (
            reward_function == 'dense' or track_pose_error
        )
        # use collision penalties
        self.apply_collision_penalties = (
            reward_function == 'dense' or apply_collision_penalties
        )
        # track per-step cube movements
        self.track_cube_step_mvmt =(
            reward_function == 'dense' or track_per_step_cube_mvmt
        )
        # reward function
        if reward_function not in ['sparse', 'dense', 'custom']:
            raise ValueError(
                f"`reward_function='{reward_function}'` is invalid."
                + " Please specify 'sparse', 'dense', or 'custom'."
            )
        self.reward_fn = reward_function
        if self.reward_fn == 'custom':
            self._custom_reward_fn = None
            self._custom_reward_fn_kwargs = {}
        
    def make(self):
        """Create the environment."""
        # initialize simulator
        super().__init__(use_GUI=self._set_GUI, gravity=self._set_gravity)
        # add robot
        self.add_robot(
            robot_controller=robot.kvG3_7_HdE_control, 
            base_position=self.robot_base_position, 
            base_orientation=self.robot_base_orientation, 
            controller_kwargs=self.robot_kwargs
        )
        # add target formation
        self.create_targets(self.target_cube_ort)
        # add physical cubes
        # self.set_cube_poses(**self.cube_pose_params)
        self.add_all_cubes()
        # extract environment attributes
        self._targets = {}
        self._cubes = {}
        for idx, env_obj in self.env_objects.items():
            env_obj_info = self.get_dict_from_repr(env_obj.__repr__())
            if env_obj_info.get('object_type') == 'virtual_cube':
                self._targets[idx] = env_obj
            if env_obj_info.get('object_type') == 'small_cube':
                self._cubes[idx] = env_obj
        self._target_ids = list(self._targets.keys())
        self._cube_ids = list(self._cubes.keys())
        self._target_face_centroid_locs = np.empty((self.num_targets, 6, 3))
        for i, t_id in enumerate(self._target_ids):
            centroid_locs = self._targets.get(t_id).get_face_centroids()
            if self.arm_control._use_lcs:
                centroid_locs = self.arm_control.pts_2_lcs(centroid_locs)
            self._target_face_centroid_locs[i, :, :] = centroid_locs
        # initialize action array
        self.__action_array = np.zeros(7)
        self.__action_dXdYdZ = self.__action_array[:3]
        self.__action_RxRyRz = self.__action_array[3:6]
        self.__action_grips = self.__action_array[-1]
        # track pose accuracy with every transition step
        if self.track_pose_error:
            self.__step_pos_goal, self.__step_ort_goal = (
                self.arm_control.get_end_eff_pose()
            )
        # track arm collisions with itself or the floor
        self._collision_counter_self = 0
        self._collision_counter_floor = 0
        # track cube distances for reward calculation
        if self.track_cube_step_mvmt:
            self._per_step_cube_dist = 0
        # episode data
        self._episode_done = False
        self._episode_sim_step_count = 0
        self._num_episodes_finished = 0
        self._episode_reward = 0
        self._total_reward = 0
        # start tracking state observations
        self.process_state()
        # track distance to cube for rewards
        if self.num_cubes == 1:
            self._start_dist_2_cube = env_utils.norm_dist_to_cube(
                self._current_ee_pos, self._cube_point_locations[0].mean(axis=0)
            )
            self._cube_dist_reward = 0

    def copy_env(self):
        """Create a replica of this environment."""
        if self.__initial_arg_vals.get('use_GUI') is True:
            raise AttributeError(
                "The environment being copied has the `use_GUI` parameter set" 
                + " to `True`. PyBullet does not allow multiple GUI clients." 
                + " To copy this environment, set `use_GUI` to `False` in the" 
                + " environment definition and try again."
            )
        replica = type(self)(**self.__initial_arg_vals.copy())
        replica_attrs = replica.__dict__.copy()
        for k in replica_attrs.keys():
            if k.endswith('__initial_arg_vals'):
                remove_key = k
        _ = replica_attrs.pop(remove_key, None)
        if not hasattr(self, '_sim_id'):
            warnings.warn(
                "*NOTE* The environment being copied was not connected (i.e. "
                + " its `make()` method was not called).\nCalling `make()` now"
                + " on the source environment. This helps ensure the copy's"
                + " initial config is as close as possible to the original.",
                stacklevel=2
            )
            self.make()
        replica.__dict__.update(
            {
                k: v for k, v in self.__dict__.items() 
                if k in replica_attrs.keys()
            }
        )
        return replica

    def custom_reward_fn(self, reward_fn:Callable, **fn_kwargs):
        """
        Define a custom reward function.

        NOTE: the instance's `.make()` method should typically be 
        called before using this method.

        keyword args:
        ------------
        reward_fn: a reward function to use
        fn_kwargs: kewyord args to pass to the function.
        """
        self._custom_reward_fn = reward_fn
        self._custom_reward_fn_kwargs = {k: v for k, v in fn_kwargs.items()}

    @property
    def targets(self):
        return self._targets
    
    @property
    def target_ids(self):
        return self._target_ids

    @property
    def cubes(self):
        return self._cubes
    
    @property
    def cube_ids(self):
        return self._cube_ids
    
    @property
    def target_locators(self):
        return self._target_face_centroid_locs

    @property
    def state_observations(self):
        return {
            'end_effector_position': self._current_ee_pos, 
            'end_effector_orientation': self._current_ee_ort, 
            'end_effector_position_error': (
                self._step_pose_error[0] if len(self._step_pose_error) > 0
                else None
            ),
            'end_effector_orientation_error': (
                self._step_pose_error[1] if len(self._step_pose_error) > 0
                else None
            ),
            'grip_finger_positions': self._current_grip_pos,  
            'grip_contact_w_cubes': self._grip_touching_cube, 
            'grip_force': self._grip_force, 
            'self_collision': self._self_collision, 
            'floor_collision': self._floor_collision, 
            'self_collision_intensity': self._collision_intensity_self, 
            'floor_collision_intensity': self._collision_intensity_floor, 
            'cube_point_locations': self._cube_point_locations, 
            'target_point_locations': self._target_point_locations, 
            'cubes_aligned_with_targets': self._cubes_aligned_w_targets, 
            'episode_done': self._episode_done, 
            'episode_reward': self._episode_reward,
            'current_episode': self._num_episodes_finished + 1, 
            'total_reward': self._total_reward
        }

    def set_cube_poses(self):
        """ 
        Specify positions and orientations of physical cubes.

        Uses polar coordinates to determine cube locations then 
        converts them to world cartesian coordinates.

        keyword args:
        ------------
        min_dist, max_dist: minimum and maximum polar distance.
        sweep: maximum polar angle to use
        delta_z_rotations: minimum and maximum amount to rotate a cube 
            about its vertical axis.
        center: center of polar coordinates (x, y). By default, the 
            robot base position is used.

        returns:
        -------
        Two arrays of length num_cubes; one for positions and another 
        for orientations.
        """
        min_dist = self.cube_pose_params.get('min_dist')
        max_dist = self.cube_pose_params.get('max_dist')
        sweep = self.cube_pose_params.get('sweep')
        delta_z_rotations = self.cube_pose_params.get('delta_z_rotations')
        center = self.cube_pose_params.get('center')

        cube_locs = np.zeros((self.num_cubes, 3), dtype=float)
        cube_orts = np.zeros((self.num_cubes, 3), dtype=float)

        coords = utils.make_polar_coords(
            min_dist, max_dist, -0.5*sweep, 0.5*sweep, n_coords=self.num_cubes
        )
        coords = utils.polar2cart(coords) + np.array(center)
        cube_locs[:, :2] = coords
        # create random z-axis rotations
        cube_orts[:, 2] = np.random.uniform(
            delta_z_rotations[0], delta_z_rotations[1], size=self.num_cubes
        )
        # convert to quaternions
        cube_orts = np.array(qtr.array.from_axis_angle(cube_orts))
        # convert to PyBullet quaternion format
        cube_orts = cube_orts[:, [1, 2, 3, 0]]

        return cube_locs, cube_orts

    def add_all_cubes(self, 
                      resolve_interferences:bool=True, 
                      max_resolution_attempts=10):
        """ 
        Adds all the physical cubes to the environment.

        If no starting poses are found, `set_cube_poses()` is called 
        with default arguments.

        If `resolve_interferences` is `True`, cubes are checked for 
        interferences every time a new cube is added and its position 
        is reassigned if interferences are detected. The process is 
        repeated for each new cube until no interferences are found or 
        `max_resolution_attempts` is reached. Resolution can take a 
        long time or may not be possible if there is a large number of 
        cubes or cube locations are tightly spaced.
        """
        if (self.starting_cube_locations is None 
            and self.starting_cube_orientations is None):
            self.starting_cube_locations, self.starting_cube_orientations = (
                self.set_cube_poses()
            )
        if self.starting_cube_locations is None:
            self.starting_cube_locations, _ = (
                self.set_cube_poses()
            )
        if self.starting_cube_orientations is None:
            _, self.starting_cube_orientations = (
                self.set_cube_poses()
            )

        center = np.array(self.cube_pose_params.get('center'))

        for i in range(self.num_cubes):
            new_obj_id = self.add_env_object(
                small_cube, 
                self.starting_cube_locations[i, :], 
                self.starting_cube_orientations[i, :]
            )
            if (resolve_interferences and i > 0):
                interferences = env_utils.check_interferences(
                    self.sim, self.env_object_ids
                )
                resolution_step = 0    # use counter to prevent long loops
                while interferences:
                    if resolution_step >= max_resolution_attempts:
                        break
                    new_obj = self.env_objects.get(new_obj_id)
                    new_loc = np.zeros(3)
                    new_coords = utils.make_polar_coords(
                        self.cube_pose_params.get('min_dist'), 
                        self.cube_pose_params.get('max_dist'), 
                        -0.5*self.cube_pose_params.get('sweep'), 
                        0.5*self.cube_pose_params.get('sweep'), 
                        n_coords=1
                    )
                    new_coords = (
                        utils.polar2cart(new_coords) + center
                    )
                    new_loc[:2] = new_coords
                    new_obj.reset(new_loc, self.starting_cube_orientations[i, :])
                    interferences = env_utils.check_interferences(
                        self.sim, self.env_object_ids
                    )
                    resolution_step += 1

    def create_targets(self, orientation_angles:Optional[Iterable]=None):
        """Add the target stack positions with virtual cubes to env."""
        if orientation_angles is None:
            orientation = (0., 0., 0., 1.)   
        else:
            if len(orientation_angles) != 3:
                raise utils.IncorrectNumberOfArgs(
                    '`orientation_angles` takes 3 elements, '
                    + f'but {len(orientation_angles)} were given.'
                )
            orientation = utils.quaternion_from_RxRyRz(*orientation_angles)
        for target_loc in self.target_coords:
            self.add_env_object(
                virtual_cube, target_loc, orientation
            )
    
    def make_new_target_formation(self, 
                                  new_formation:np.array, 
                                  new_position:Optional[Iterable]=None, 
                                  new_orientation:Optional[Iterable]=None):
        """ 
        Replace the current target formation with a new one.

        A new centroid position and cube orientation can also be 
        specified.
        """
        for target_id in self.target_ids:
            _ = self.remove_env_object(target_id)
        self.target_formation = 'specified'
        self.num_targets = len(new_formation)
        if new_position:
            self.target_ctrd_pos = new_position
        if new_orientation:
            orientation = new_orientation
        else:
            orientation = self.target_cube_ort
        self.target_coords = new_formation + np.array(self.target_ctrd_pos)
        self.create_targets(orientation)

    def get_cube_face_centroid_coords(self):
        """ 
        Returns the current face centroid coordinates of all cubes.
        """
        cube_face_centroid_coords = np.empty((self.num_cubes, 6, 3))
        for i, c_id in enumerate(self._cube_ids):
            centroid_coords = self._cubes.get(c_id).get_face_centroid_locs()
            if self.arm_control._use_lcs:
                centroid_coords = self.arm_control.pts_2_lcs(centroid_coords)
            cube_face_centroid_coords[i, :, :] = centroid_coords

        return cube_face_centroid_coords

    def current_joint_positions(self):
        """Returns the current angles of all joints."""
        return np.concatenate(
            (
                self.arm_control.arm_joint_positions(), 
                self.arm_control.grip_finger_positions()
            )
        )

    def get_cube_movement(self, before, after):
        """
        Get the distance moved by the cubes in a step.
        
        Calculates the Euclidean distance between the cube locations 
        before and after a transition step. This can be used for 
        shaping dense rewards.

        before, after: cube point locations before and after the step.
        """
        return np.linalg.norm(after - before)
    
    def check_target_alignment(self, tolerance:float=1e-3):
        """
        Check if any cube is aligned with a target.
        
        Returns an array with `True` values for all targets that have 
        been satisfied.

        tolerance: target is considered aligned if coordinates are 
            within this value.
        """
        target_intersection_pts = np.empty((self.num_targets, 6, 3))
        targets_aligned = np.array([False]*self.num_targets, dtype=bool)
        for i, t_id in enumerate(self.target_ids):
            target = self.targets.get(t_id)
            target_coords = target.get_face_centroids()
            ray_intersections = target.detect_object_placement()
            target_intersection_pts[i, :, :] = ray_intersections
            targets_aligned[i] = np.allclose(
                target_coords, ray_intersections, atol=tolerance
            )
        return target_intersection_pts, targets_aligned

    def detect_floor_collision(self) -> bool:
        floor_collision = self._sim.getContactPoints(
            bodyA=self._robot_id, bodyB=self._floor_surface
        )
        return True if len(floor_collision) > 0 else False

    def episode_timed_out(self) -> bool:
        if self._episode_sim_step_count >= self._episode_sim_step_limit:
            return True
        return False

    def reset(self, reset_to_random:bool=False):
        """
        Resets the environment.

        If the robot's base pose was initialized as 'random' and 
        `reset_to_random` is `True`, the robot base is reset to a random 
        new pose.
        """
        if reset_to_random and self._robot_pose_init == 'random':
            self.robot_base_position = (
                np.random.uniform(*self.rand_pos_lims), 
                np.random.uniform(*self.rand_pos_lims), 
                0.
            )
            self.robot_base_orientation = utils.quaternion_from_RxRyRz(
                0., 0., np.random.uniform(*self.rand_ort_lims)
            )
            self.reset_robot(
                self.robot_base_position, self.robot_base_orientation
            )
        else:
            self.reset_robot()
        for cube_id in self.cube_ids:
            self.reset_env_object(cube_id)
        self._episode_done = False
        self._episode_sim_step_count = 0
        self._episode_reward = 0
        self._collision_counter_self = 0
        self._collision_counter_floor = 0
        self._num_episodes_finished += 1
        self.process_state()

    def process_state(self):
        self._current_ee_pos, self._current_ee_ort = (
            self.arm_control.get_end_eff_pose()
        )
        if self.track_pose_error:
            self._step_pose_error = utils.calculate_distance_error(
                self.__step_pos_goal, self.__step_ort_goal, 
                self._current_ee_pos, self._current_ee_ort
            )
        else:
            self._step_pose_error = ()
        self._current_grip_pos = self.arm_control.grip_finger_positions()
        self._grip_touching_cube = np.array(
            [self.arm_control.detect_grip_contact(c) for c in self.cube_ids], 
            dtype=bool
        )
        self._grip_force = self.arm_control.get_grip_force()
        self._self_collision = self.arm_control.detect_self_collision()
        self._floor_collision = self.detect_floor_collision()
        # collision intensity = 
        # (number of collisions per transition)/(number of sim substeps)
        self._collision_intensity_self = (
            self._collision_counter_self/self._n_substeps
        )
        self._collision_intensity_floor = (
            self._collision_counter_floor/self._n_substeps
        )
        self._cube_point_locations = self.get_cube_face_centroid_coords()
        self._target_point_locations = self.target_locators
        _, self._cubes_aligned_w_targets = (self.check_target_alignment())

    def calculate_reward(self):
        """ 
        Calculate episode transition step rewards.

        If a reward function is given, it is used for calculation. 
        Otherwise, a default reward is calculated based on elapsed 
        time (in timesteps), cube alignment success, and pose and 
        collision penalties if applicable.

        **fn_kwargs: keyword args to pass to the reward function if used.
        """
        if self.reward_fn == 'custom':
            if self._custom_reward_fn is None:
                raise AttributeError(
                    "Reward fn was set to 'custom' but no custom function was"
                    + " defined. You can use the `custom_reward_fn()` method"
                    + " to define one, or set the reward function to either " 
                    + "'sparse' or 'dense'."
                )
            reward = self._custom_reward_fn(**self._custom_reward_fn_kwargs)
        elif self.reward_fn == 'sparse':
            # rewards are scaled to [-1, 1]
            if np.all(self._cubes_aligned_w_targets):
                reward = 1.
            else:
                reward = -1. / self._max_transition_steps
        else:
            if self.track_pose_error:
                pose_error_penalty = env_utils.apply_pose_penalty(
                    self._step_pose_error[0], self._step_pose_error[1]
                )*(-1)
            else:
                pose_error_penalty = 0
            if self.apply_collision_penalties:
                collision_penalties  = (
                    0.5*self._collision_intensity_self 
                    + 0.5*self._collision_intensity_floor
                )*(-1)
            else:
                collision_penalties = 0
            # add a reward when both gripper fingers touch a cube
            touching_cube = np.sum(np.all(self._grip_touching_cube, axis=1))
            # add a reward for moving a cube
            moved_cube = (touching_cube*self._per_step_cube_dist)/self.num_cubes
            # reward for cube alignment, scaled over n targets
            alignment_success = (
                np.sum(self._cubes_aligned_w_targets)/self.num_targets
            )
            gripper_2_cube_reward = self._cube_dist_reward if self.num_cubes == 1 else 0
            # assign -1 for every timestep the final task is not achieved and 
            # calculate overall reward, scaled by episode time limit
            reward = (
                -1 + pose_error_penalty*0.1 + collision_penalties + touching_cube
                + moved_cube + alignment_success + gripper_2_cube_reward
            )/self._max_transition_steps
        return reward

    def run_actions(self, actions:np.array, sleep:Optional[float]=None):
        """ 
        Run the given actions and step through the episode.

        keyword args:
        ------------
        actions: array of actions to take in the transition step.
        sleep: adds a sleep interval between simulation step to slow 
            down the motion in GUI mode. 
            Ignored if `use_GUI` is `False`.
        """
        self.__action_array[self._available_actions] = actions
        sequence, self.__step_pos_goal, self.__step_ort_goal = (
            self.arm_control.get_actuator_sequence(
                dx_dy_dz=self.__action_dXdYdZ, 
                dRx_dRy_dRz=self.__action_RxRyRz, 
                delta_grip=self.__action_grips, 
                num_steps=self._n_substeps
            )
        )
        self._collision_counter_self = 0
        self._collision_counter_floor = 0
        if self.track_cube_step_mvmt:
            cubes_before = self._cube_point_locations
        # track dist from gripper to cube for rewards
        # only applies to pick and place tasks
        if self.num_cubes == 1:
            dist_2_cube_before = env_utils.norm_dist_to_cube(
                self._current_ee_pos, self._cube_point_locations[0].mean(axis=0)
            )
        collision_stop = False
        for step, setting in enumerate(sequence):
            if step == 0:
                prev_jt_posns = self.current_joint_positions()
            if collision_stop:
                if step == len(sequence) - 1:
                    self.arm_control.set_actuators(prev_jt_posns)
                    self.simulation_step(sleep=sleep)
                    collision_stop = False
                else:
                    if self_coll:
                        self._collision_counter_self += 1
                    if floor_coll:
                        self._collision_counter_floor += 1
                    self.simulation_step(sleep=sleep)
            else:
                self.arm_control.set_actuators(setting)
                self.simulation_step(sleep=sleep)
                self_coll = self.arm_control.detect_self_collision()
                floor_coll = self.detect_floor_collision()
            if self_coll or floor_coll:
                collision_stop = True
            if not collision_stop:
                prev_jt_posns = sequence[step]
            self._episode_sim_step_count += 1 
        self.process_state()
        if self.track_cube_step_mvmt:
            cubes_after = self._cube_point_locations
            self._per_step_cube_dist = self.get_cube_movement(
                cubes_before, cubes_after
            )
        if self.num_cubes == 1:
            dist_2_cube_after = env_utils.norm_dist_to_cube(
                self._current_ee_pos, self._cube_point_locations[0].mean(axis=0)
            )
            self._cube_dist_reward = env_utils.get_dist_reward(
                dist_2_cube_before, dist_2_cube_after, self._start_dist_2_cube
            )
        reward = self.calculate_reward()
        self._episode_reward += reward
        self._total_reward += reward
        if (np.all(self._cubes_aligned_w_targets)
            or
            self.episode_timed_out()):
            self._episode_done = True
        if (self._episode_done and self._reset_on_episode_end):
            self.reset()

    def close(self):
        self.process_state()
        self._connection.close()

