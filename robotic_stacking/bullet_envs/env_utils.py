from dataclasses import dataclass, field
from itertools import product
from typing import Iterable, Tuple

import numpy as np

from robotic_stacking import utils

# ----------------------------------------------------------------------------
# Utility functions and classes for setting up and interacting with a 
# PyBullet environment.
# ----------------------------------------------------------------------------

def calculate_simulation_substeps(transition_steps_per_sec:int, 
                                  PyBullet_timestep:int=240) -> int:
    """
    Calculate the number of substeps in a transition step.
    
    transition_steps_per_sec: the number of transition steps taken 
        by an agent per second. One transition step can have several 
        simulation substeps.
    PyBullet_timestep: the simulation timestep in Hz. This is the 
        number of times per second PyBullet updates the physics 
        server state. PyBullet default is 240 Hz and it is recommended  
        by the documentation to leave at that for most cases. 
    """
    return PyBullet_timestep // transition_steps_per_sec


def check_interferences(simulation, env_object_ids) -> set:
    """Check for interfering objects in the simulation."""
    simulation.performCollisionDetection()
    contact_pts = simulation.getContactPoints()
    interfering_bodies =  [
            (b[1], b[2]) for b in contact_pts 
            if (b[1] in env_object_ids and b[2] in env_object_ids)
    ]
    if len(interfering_bodies) > 0:
        return set(interfering_bodies)
    else:
        return None


def target_aligned(target_points, 
                   rays_from, rays_to, 
                   target_ID, 
                   tolerance=1e-3) -> bool:
    """
    Check if an item is aligned with a target using a ray test.

    Keyword args:
    -------------
    target_points: (numpy array) target points for checking alignment. 
        Can be obtained by calling the function `get_target_points()`.
    rays_from; rays_to: each is a numpy array of the starting points 
        and endpoints respectively of the ray-test rays. Obtained by 
        calling `define_rays_from_to()`.
    target_ID: the target location for alignment.
    tolerance: alignment tolerance in meters.

    Returns:
    --------
    `True` if an item is aligned with the target.
    """
    ray_hits = utils.get_points_from_rays(rays_from, rays_to, target_ID)
    ray_hits = ray_hits.view('float').reshape(target_points.shape)

    return np.allclose(target_points, ray_hits, atol=tolerance)


def apply_pose_penalty(position_error:float, orientation_error:float, 
                       position_tol:float=1e-3, 
                       orientation_tol:float=0.01*np.pi, 
                       max_position_error:float=0.025, 
                       position_weight:float=0.5) -> float:
    """
    Calculate penalty relative to end-effector deviation from goal.

    keyword args:
    ------------
    position_error, orientation_error: distance errors returned by
        `utils.calculate_distance_error()`.
    position_tol, orientation_tol: tolerance values for position and 
        orientation. No penalty is calculated for values within the 
        tolerances.
    max_position_error: maximum error allowed for position error. The 
        penalty for a given position error is normalized to this max 
        value. Error values greater than `max_position_error` are 
        clipped to 1.
    position_weight: apply a weight between [0., 1.] to the position 
        error. The weight of the orientation error is calculated as 
        (1 - position_weight).

    returns:
    -------
    A single scalar penalty value for the deviation from the desired 
    position and orientation.
    """
    if (position_weight > 1. or position_weight < 0.):
        raise ValueError('`position_weight` must be between 0. and 1.')
    ort_weight = 1. - position_weight
    if position_error <= position_tol:
        pos_penalty = 0.
    else:
        pos_penalty = (
            np.min([position_error, max_position_error])/max_position_error
        )
    if orientation_error <= orientation_tol:
        ort_penalty = 0.
    else:
        ort_penalty = orientation_error / np.pi

    return position_weight*pos_penalty + ort_weight*ort_penalty

# ----------------------------------------------------------------------------


@dataclass
class target_formation:
    """
    Create a structure to set up targets in a stacking environment.

    The target formation is a 3D coordinates matrix indexed by 
    positions. For example, if the target structure is 
    8 cubes long x 8 cubes wide x 8 cubes high, 
    `target_formation[0, 1, 2]` returns coordinates corresponding to
    the first length position, second width position and 
    third level high.

    keyword args:
    ------------
    target_cube_dims: size of the cubes being stacked.
    stack_length_X: number of cubes in the length (x) dimension.
    stack_width_Y: number of cubes in the width (y) dimension.
    stack_height_Z: number of cubes in the height (z) dimension.
    """
    target_cube_dims: Tuple[float, float, float] = (0.06, 0.06, 0.06)
    stack_length_X: int = 8
    stack_width_Y: int = 8
    stack_height_Z: int = 8
    coords_X: np.array = field(init=False)
    coords_Y: np.array = field(init=False)
    coords_Z: np.array = field(init=False)
    coords_3D: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        x_min = (
            -(self.stack_length_X/2)*self.target_cube_dims[0] 
            + self.target_cube_dims[0]/2
        )
        x_max = (
            (self.stack_length_X/2)*self.target_cube_dims[0] 
            - self.target_cube_dims[0]/2
        )
        y_min = (
            -(self.stack_width_Y/2)*self.target_cube_dims[0] 
            + self.target_cube_dims[0]/2
        )
        y_max = (
            (self.stack_width_Y/2)*self.target_cube_dims[0] 
            - self.target_cube_dims[0]/2
        )
        z_min = 0.
        z_max = (
            (self.stack_height_Z)*self.target_cube_dims[0] 
            - self.target_cube_dims[0]
        )
        self.coords_X = np.linspace(x_min, x_max, num=self.stack_length_X)
        self.coords_Y = np.linspace(y_min, y_max, num=self.stack_width_Y)
        self.coords_Z = np.linspace(z_min, z_max, num=self.stack_height_Z)
        self.coords_3D = np.stack(
            np.meshgrid(self.coords_X, self.coords_Y, self.coords_Z), 
            axis=-1
        )

    def create_target_formation(self, location_indices:Iterable) -> np.ndarray:
        """
        Get coordinates of the given locations in the target structure.

        Returns an array of 3D coordinates corresponding to the given 
        `location_indices`.
        """
        target_cube_locs = np.empty((len(location_indices), 3))
        for row, loc in enumerate(location_indices):
            target_cube_locs[row] = self.coords_3D[loc[0], loc[1], loc[2]]
        return target_cube_locs


def four_corner_structure(world_position:Iterable, 
                          n_cubes:int=8, 
                          cube_spacing:int=2) -> np.ndarray:
    """
    Define a structure with cubes stacked at four corners.

    The four stacks are equal in size and symmetrically placed relative 
    to a centroid.

    keyword args:
    ------------
    world_position: a list, tuple or array of the centroid location of 
        the four stacks in world coordinates.
    n_cubes: number of cubes to stack.
    cube_spacing: the number of cube spaces between two corner cubes
        (e.g. a cube spacing of 2 means you can fit 2 cubes between 
        corner stacks).

    returns:
    -------
    Array of coordinates corresponding to target positions.
    """
    if n_cubes % 4 != 0:
        raise ValueError(
            "This formation requires `n_cubes` to be a multiple of 4."
        )
    stack_x, stack_y = cube_spacing + 2, cube_spacing + 2
    height = n_cubes // 4
    t_formation = target_formation(
        stack_length_X=stack_x, stack_width_Y=stack_y, stack_height_Z=height
    )
    level_indices = [[0, 0], [0, -1], [-1, 0], [-1, -1]]
    loc_indices = []
    for level in range(height):
        for level_idx in level_indices: 
            loc_indices.append(level_idx + [level])
    target_locations = (
        t_formation.create_target_formation(loc_indices) + np.array(world_position)
    )
    return target_locations


def simple_pyramid_structure(world_position:Iterable, 
                             level_sizes:Iterable=[3, 2, 1]) -> np.ndarray:
    """
    Define a pyramid structure.

    keyword args:
    ------------
    world_position: a list, tuple or array of the centroid location of 
        the pyramid in world coordinates.
    level_sizes: 2D size of each level side. For example, a [3, 2, 1] 
        structure would have a 3x3=9 base, 2x2=4 for the next level 
        and so on.

    returns:
    -------
    Array of coordinates corresponding to target positions.
    """
    target_positions = {}
    for level, size in enumerate(level_sizes):
        t_formation = target_formation(
            stack_length_X=size, stack_width_Y=size, stack_height_Z=1
        )
        Z_pos = level * t_formation.target_cube_dims[2]
        target_coords = (
            t_formation.create_target_formation(
                [l_idx for l_idx in product(range(size), range(size), [0])]
            ) 
            + np.array(world_position) 
            + np.array([0., 0., Z_pos])
        )
        target_positions[f'level{level}'] = target_coords

    return np.concatenate(list(target_positions.values()))
