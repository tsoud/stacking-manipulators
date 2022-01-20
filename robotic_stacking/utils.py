from typing import List, Tuple, Union

import numpy as np
import pybullet as pbt
import quaternionic as qtr

# ----------------------------------------------------------------------------
# General utility functions to use across scripts
# ----------------------------------------------------------------------------


class IncorrectNumberOfArgs(TypeError):
    """
    Raise if number of arguments to a function is incorrect.

    This can happen often when switching between lists, tuples and
    Numpy arrays, or when converting between angle and quaternion
    representations.
    """
    pass


def getRobotBasePose(simulation, robot_ID:int, 
                     AxisAngle:bool=False) -> Tuple[Tuple, Tuple]:
    """
    Get the robot base position and orientation.

    This function is helpful for determining the robot's local 
    coordinate system for applying transformations.

    returns:
    -------
    two tuples: tuple of X, Y, Z translations from world origin and 
        a tuple of orientation in either (x, y, x, w) quaternions or 
        (x, y, z, theta) if `AxisAngle` is True.
    """
    pose = simulation.getBasePositionAndOrientation(robot_ID)
    posn, ornt = pose[0], pose[1]
    if AxisAngle:
        ornt = pbt.getAxisAngleFromQuaternion(ornt)
    return posn, ornt


def transform_CS(target_pos, target_ort, 
                 transformation:Tuple) -> Tuple[Tuple, Tuple]:
    """
    Transform a pose from one coordinate system to another.

    This function is especially useful for translating from the 
    simulation's world coordinates to local coordinates of a robot.

    keyword args:
    ------------
    target_pos, target_ort: the target position and orientation to 
        transform, respectively.
    transformation: a tuple with a position vector and orientation 
        quaternion representing the transformation matrix between 
        coordinate systems.
    """
    return pbt.multiplyTransforms(
        transformation[0], transformation[1], target_pos, target_ort
    )


def pyb_quaternion(q_in:Union[np.array, List[float], Tuple[float]], 
                   reverse=False):
    """
    Re-order a quaternion so it conforms to PyBullet's convention.

    Most libraries use the convention (w, x, y, z) to represent 
    quaternions but PyBullet uses (x, y, z, w). This function helps 
    when using other libraries like Scipy, pyquaternion or 
    roboticstoolbox for calculations.

    keyword args:
    ------------
    input_quaternion: the quaternion to reorder.
    reverse: if `True`, assumes input is in PyBullet's (x, y, z, w) 
        convention and desired output is (w, x, y, z). Otherwise the 
        default is (x, y, z, w) --> (w, x, y, z).
    """
    if not len(q_in) == 4:
        raise IndexError('Input quaternion should have four elements.')
    if isinstance(q_in, (list, tuple)):
        q_in = np.array(q_in, dtype=float)
    if reverse:
        return q_in[[3, 0, 1, 2]]
    else:
        return q_in[[1, 2, 3, 0]]


def q_interp(start_ort:Union[np.array, List[float], Tuple[float]], 
             target_ort:Union[np.array, List[float], Tuple[float]], 
             num_steps:int=50, use_pyb_order:bool=True):
    """
    Quaternion interpolation between two orientations.

    keyword args:
    ------------
    start_ort, target_ort: starting and target orientations.
    num_steps: number of interpolation steps including start and end.
    use_pyb_order: use PyBullet's (x, y, z, w) quaternion format for 
        input and results.
    """
    if isinstance(start_ort, (list, tuple)):
        start_ort = np.array(start_ort, dtype=float)
    if isinstance(target_ort, (list, tuple)):
        target_ort = np.array(target_ort, dtype=float)
    if use_pyb_order:
        start_ort = pyb_quaternion(start_ort, reverse=True)
        target_ort = pyb_quaternion(target_ort, reverse=True)
    start, end = qtr.array(start_ort), qtr.array(target_ort)
    s = np.linspace(0, 1, num=num_steps, endpoint=True)
    # Use quaternionic library fn (qtr.slerp) for speed
    q_stops = np.array(qtr.slerp(start, end, s))
    return q_stops[:, [1, 2, 3, 0]] if use_pyb_order else q_stops


def quaternion_from_RxRyRz(Rx, Ry, Rz, pyb_format=True):
    """
    Get a quaternion from Rx, Ry, Rz rotations.

    If `pyb_format` is True, the quaternion is returned in PyBullet's 
    [x, y, z, w] convention.
    """
    q_rot = qtr.array.from_axis_angle(
        [[Rx, 0., 0.], [0., Ry, 0.], [0., 0., Rz]]
    )
    q_rot = np.array(q_rot[0]*q_rot[1]*q_rot[2])
    if pyb_format:
        q_rot = q_rot[[1, 2, 3, 0]]
    return q_rot


def increment_RxRyRz(current_ort, dRx_dRy_dRz):
    """Calculate a new orientation from axis-angle increments."""
    dRx, dRy, dRz = dRx_dRy_dRz[0], dRx_dRy_dRz[1], dRx_dRy_dRz[2]
    q_rot = qtr.array.from_axis_angle(
        [[dRx, 0., 0.], [0., dRy, 0.], [0., 0., dRz]]
    )
    q_rot = q_rot[0]*q_rot[1]*q_rot[2]
    q_curr = qtr.array(current_ort[[3, 0, 1, 2]])
    return np.array(q_curr*q_rot)[[1, 2, 3, 0]]


def calculate_grip_stops(current_finger_pos:np.array, 
                         delta:np.array, n_steps:int):
    """Calculate incremental stops to move grippers by `delta`."""
    deltas = np.linspace(np.array([0., 0.]), delta, n_steps)
    curr_pos = np.tile(current_finger_pos, (n_steps, 1))
    return curr_pos + deltas


def calculate_max_joint_deltas(max_joint_velocities:Union[
                                    List[float], np.array
                                    ], 
                               simulation_steps_per_sec:int):
    """
    Calculate the maximum amount joints can move in a simulation step.

    The maximum amount a joint can change position for any step is 
    limited by the maximum speed its actuator motor can travel. This 
    function calculates the maximum amount based on the simulation 
    step size.

    keyword args:
    ------------
    max_joint_velocities: array or array-like; maximum velocities of all the 
        joints being used in the simulation. Assumes rad/s for rotational 
        joints and m/s for linear joints. These are properties of the robot 
        and gripper.
    simulation_steps_per_sec: number of time steps per second used by the 
        simulator to update the environment dynamics.

    For example, if PyBullet's default simulation step size of 1/240 (240 Hz 
    or 240 steps/sec) is used and the maximum rotational speed of a joint's 
    actuator motor is 1.0 rad/s, the maximum angle this joint can rotate per 
    simulation step is about +/-(1.0 / 240) = +/- 0.004167 rad.
    """
    max_velocities = np.asarray(max_joint_velocities, dtype=float)
    return max_velocities / simulation_steps_per_sec


def calculate_distance_error(goal_position:np.array, goal_orientation:np.array, 
                             actual_position:np.array, actual_orientation:np.array):
    """
    Measures the error between desired and actual pose.

    This function uses Euclidean distance to measure the translational  
    (position) error and the geodesic distance on the unit sphere 
    for the rotational (orientation) error.

    returns:
    -------
    A scalar distance value in meters for position and a scalar 
    distance value between [0, pi) in radians for orientation. 
    """
    # rearrange input quaternions and convert to quaternionic arrays
    goal_orientation = qtr.array(
        pyb_quaternion(goal_orientation, reverse=True)
    )
    actual_orientation = qtr.array(
        pyb_quaternion(actual_orientation, reverse=True)
    )
    position_err = np.linalg.norm(goal_position - actual_position)
    orientation_err = qtr.distance.rotation.intrinsic(
        goal_orientation, actual_orientation
    )
    return position_err, orientation_err


def define_rays_from_to(dX_dY_dZ:Union[List, Tuple, np.array], ends=None):
    """
    Defines 'from' and 'to' points for ray-test queries on a body.

    Currently, this function supports only cube and cuboid objects 
    with 6 faces.

    Keyword args:
    -------------
    dX_dY_dZ: offset along each axis to determine where rays start.
        Rays are directed from outside the cube toward cube's center 
        of mass. There are six rays total, since each axis will have a
        positive and negative ray.
        The list should have one or three elements. If one element is 
        given, it is assumed the same value applies in all directions.
    ends: ray endpoints. If not given, origin (0, 0, 0) is assumed.
        Takes a numpy array or list of lists (e.g. [[0, 0, 0], ...]).

    Returns:
    --------
    Two numpy arrays of shape (6, 3) corresponding to the starting 
    points and endpoints of the rays.
    """
    if len(dX_dY_dZ) == 1:
        dX_dY_dZ = np.array(list(dX_dY_dZ) * 3)
    else:
        dX_dY_dZ = np.array(dX_dY_dZ)
    rays_from = np.concatenate(
        (np.eye(3, 3)*dX_dY_dZ, (-1)*np.eye(3, 3)*dX_dY_dZ)
    )
    rays_to = np.array(ends) if ends else np.zeros([6, 3])

    return rays_from, rays_to


def get_points_from_rays(simulation, rays_from, rays_to, body_id):
    """
    Performs a raycast and converts the ray intersections to points.

    This function can be used to track the points on a body as it 
    moves through the environment or to determine if an object is
    being placed at a target location.

    Keyword args:
    -------------
    rays_from: starting points of each ray (numpy array or 
        list of tuples).
    rays_to: endpoints of each ray (numpy array or list of tuples).
    body_id: the rays should be tied to a unique object ID. For a 
        physical object (e.g. a cube to pick and place), the rays will 
        track the points at the centroids of its faces. For a virtual 
        target, the rays can be used to detect object alignment.
    simulation_ID: the simulation with which the body is tied (when 
        running more than one simulation at a time).

    Returns:
    --------
    A structured numpy array of x, y, z coordinates from a batch of 
    ray intersections.
    """
    ray_hit_info = simulation.rayTestBatch(
        rayFromPositions=rays_from, 
        rayToPositions=rays_to,
        parentObjectUniqueId=body_id, 
        parentLinkIndex=-1
    )
    point_locs = np.array(
        [p[3] for p in ray_hit_info], dtype=float
    )
    return point_locs


def make_polar_coords(min_r, max_r, 
                      min_theta, max_theta, 
                      n_coords) -> np.array:
    r_vals = np.random.uniform(min_r, max_r, size=n_coords)
    thetas = np.random.uniform(min_theta, max_theta, size=n_coords)
    return np.concatenate(([r_vals], [thetas])).T


def polar2cart(polar_coords) -> np.array:
    cart_coords = np.zeros_like(polar_coords)
    cart_coords[:, 0] = polar_coords[:, 0] * np.cos(polar_coords[:, 1])
    cart_coords[:, 1] = polar_coords[:, 0] * np.sin(polar_coords[:, 1])
    return cart_coords
