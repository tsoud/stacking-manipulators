import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pybullet as pbt
import pybullet_data
import quaternionic as qtr
import spatialmath.base as smb
from pybullet_utils import bullet_client as bc
from spatialmath import SE3

from robotic_stacking import assets, utils

# ----------------------------------------------------------------------------
# Non-robot objects to use in the simulation environment.
# ----------------------------------------------------------------------------

# Base Classes
# ------------

class physical_object(ABC):
    """An object with mass and collision properties."""

    @abstractmethod
    def __init__():
        pass

    @property
    @abstractmethod
    def sim(self):
        """Return the simulator where the object is used."""
        pass
    
    @property
    @abstractmethod
    def env_id(self):
        """Return the unique ID of the object in the simulator."""
        pass

    @property
    @abstractmethod
    def mass(self):
        pass

    @property
    @abstractmethod
    def dimensions(self):
        pass

    @abstractmethod
    def get_pose(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def delete(self):
        pass


class virtual_object(ABC):
    """
    A virtual object without mass or collision properties.
    
    Virtual objects can be used to identify and locate target poses 
    and check alignment with physical objects. Virtual objects are
    usually stationary.
    """

    @abstractmethod
    def __init__():
        pass

    @property
    @abstractmethod
    def sim(self):
        """Return the simulator where the object is used."""
        pass

    @property
    @abstractmethod
    def env_id(self):
        """Return the unique ID of the object in the simulator."""
        pass

    @property
    @abstractmethod
    def dimensions(self):
        pass

    @abstractmethod
    def get_pose(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def delete(self):
        pass

# ----------------------------------------------------------------------------

class small_cube:
    """ 
    A small cube for the robot to manipulate in the environment.

    simulation: the PyBullet simulation where the cube will be used.
    position: x, y, z initial position. Z position (perpendicular to 
        floor) is automatically adjusted for cube height.
    orientation: initial angles (radians) about x, y, z axes.
    maximal_coords: a PyBullet option that can improve performance 
        when set to `True`. It can also cause issues with object 
        removal, so object removal is disabled when this option is 
        used.
    """
    
    def __init__(self, 
                 simulation, 
                 position:Union[Tuple, List]=(0., 0., 0.), 
                 orientation:Union[Tuple, List]=(0., 0., 0., 1.), 
                 maximal_coords=True):
        self.__urdf_path = assets.find_urdf_objects().get('small_cube')
        tree = ET.parse(self.__urdf_path)
        self.__urdf_root = tree.getroot()
        for child in self.__urdf_root:
            mass = child.find('./inertial/mass')
            mass = float(mass.attrib.get('value'))
            dimensions = child.find('./collision/geometry/box')
            dimensions = str(dimensions.attrib.get('size'))
        self._mass = mass
        self._dimensions = [float(d) for d in dimensions.split(' ')]
        # tie to simulation
        self._sim = simulation
        # adjust the position for cube height
        self.ht_adj = self.dimensions[-1]/2
        self._init_pos = (position[0], position[1], position[2] + self.ht_adj)
        # self._init_ort = utils.quaternion_from_RxRyRz(*orientation)
        self._init_ort = orientation
        # place in environment
        self._env_id = self._sim.loadURDF(
            self.__urdf_path,
            basePosition=self._init_pos,
            baseOrientation=self._init_ort, 
            useMaximalCoordinates=maximal_coords
        )
        self._use_maximal = maximal_coords
        self.__obj_info = namedtuple(
            'obj_info', 'object_type, initial_position, initial_orientation')
        # set up rays for object tracking
        self._rays_from, self._rays_to = utils.define_rays_from_to(
            np.array(self._dimensions)*1.05
        )

    def __repr__(self):
        obj_info = self.__obj_info(
            self.__class__.__name__, self._init_pos, self._init_ort
        )
        return str(obj_info)

    @property
    def sim(self):
        """Return the simulator where the object is used."""
        return self._sim
    
    @property
    def env_id(self):
        """Returns the cube's unique ID in the environment."""
        return self._env_id

    @property
    def mass(self):
        return self._mass

    @property
    def dimensions(self):
        return self._dimensions

    def get_pose(self) -> Tuple[np.array, np.array]:
        """Returns current position and orientation"""
        pose = self._sim.getBasePositionAndOrientation(self._env_id)
        return np.array(pose[0]), np.array(pose[1])    # position, orientation

    def reset(self, 
              new_position:Optional[Union[List, Tuple]]=None, 
              new_orientation:Optional[Union[List, Tuple]]=None):
        """
        Reset position and orientation in current environment.
        
        If a new position and/or orientation is given, the cube will 
        reset to the new pose. Otherwise, it will reset to its original 
        pose when first created.
        """
        if new_position is None:
            pos = self._init_pos
        else:
            # remember to adjust for object height
            pos = (
                new_position[0], new_position[1], new_position[2] + self.ht_adj
            )
        ort = self._init_ort if new_orientation is None else new_orientation
        self._sim.resetBasePositionAndOrientation(self._env_id, pos, ort)
        self._init_pos = pos
        self._init_ort = ort
    
    def delete(self) -> bool:
        """Remove the body from the environment."""
        if self._use_maximal:
            print('Info:`maximal_coords` set to `True`, resetting instead...')
            self.reset()
            return False
        else:
            self._sim.removeBody(self._env_id)
            return True

    def get_face_centroid_locs(self):
        """Track face centroid locations in the simulation."""
        return utils.get_points_from_rays(
            self.sim, self._rays_from, self._rays_to, self._env_id
        )
    

# ----------------------------------------------------------------------------

class virtual_cube(virtual_object):
    """ 
    A virtual cube representing target poses for stacking real cubes.

    The virtual cube is stationary.

    simulation: the PyBullet simulation where the cube will be used.
    position: x, y, z initial position. Z position (perpendicular to 
        floor) is automatically adjusted for cube height.
    orientation: initial angles (radians) about x, y, z axes.
    """
    
    def __init__(self, 
                 simulation, 
                 position=np.array([0., 0., 0.]), 
                 orientation=np.array([0., 0., 0.])):
        self.__urdf_path = assets.find_urdf_objects().get('fake_cube')
        tree = ET.parse(self.__urdf_path)
        self.__urdf_root = tree.getroot()
        for child in self.__urdf_root:
            dimensions = child.find('./visual/geometry/box')
            dimensions = str(dimensions.attrib.get('size'))
        self._dimensions = [float(d) for d in dimensions.split(' ')]
        # tie to simulation
        self._sim = simulation
        # adjust the position for cube height
        self.ht_adj = self.dimensions[-1]/2
        self._init_pos = position + np.array([0., 0., self.ht_adj])
        self._init_ort = utils.quaternion_from_RxRyRz(*orientation)
        # place in environment
        self._env_id = self._sim.loadURDF(
            self.__urdf_path,
            basePosition=self._init_pos,
            baseOrientation=self._init_ort, 
            useFixedBase=True
        )
        self.__obj_info = namedtuple(
            'obj_info', 'object_type, initial_position, initial_orientation')
        # set up rays to detect object placement
        self._rays_from, self._rays_to = utils.define_rays_from_to(
            np.array(self._dimensions)*1.10
        )

    def __repr__(self):
        obj_info = self.__obj_info(
            self.__class__.__name__, self._init_pos, self._init_ort
        )
        return str(obj_info)

    @property
    def object_type(self):
        return self.__class__.__name__
    
    @property
    def sim(self):
        """Return the simulator where the object is used."""
        return self._sim

    @property
    def env_id(self):
        """Returns the cube's unique ID in the environment."""
        return self._env_id

    @property
    def dimensions(self):
        return self._dimensions

    def get_pose(self) -> Tuple[np.array, np.array]:
        """Returns current position and orientation"""
        pose = self._sim.getBasePositionAndOrientation(self._env_id)
        return np.array(pose[0]), np.array(pose[1])    # position, orientation

    def reset(self, 
              new_position:Optional[Union[List, Tuple]]=None, 
              new_orientation:Optional[Union[List, Tuple]]=None):
        """
        Reset position and orientation in current environment.
        
        If a `new_pose` is given, the cube will reset to this new 
        pose. Otherwise, it will reset to its original pose when 
        created.
        """
        if new_position is None:
            pos = self._init_pos
        else:
            # remember to adjust for object height
            pos = (
                new_position[0], new_position[1], new_position[2] + self.ht_adj
            )
        ort = self._init_ort if new_orientation is None else new_orientation
        self._sim.resetBasePositionAndOrientation(self._env_id, pos, ort)
        self._init_pos = pos
        self._init_ort = ort
    
    def delete(self):
        """Remove the body from the environment."""
        self._sim.removeBody(self._env_id)

    def get_face_centroids(self) -> np.array:
        """
        Get coordinates of the centroids of the virtual cube's faces.

        The face centroids of the virtual target can be used to align 
        the physical cubes.

        Returns:
        --------
        A numpy array of shape (6, 3) of the face centroid locations   
        in x, y, z world coordinates.
        """
        dX_dY_dZ = np.array(self.dimensions)/2
        pts = np.concatenate(
            (np.eye(3, 3)*dX_dY_dZ, (-1)*np.eye(3, 3)*dX_dY_dZ)
            )
        # Get position and orientation
        tgt_pos, tgt_ort = self.get_pose()
        # Get the rotation matrix from the orientation
        tgt_R = pbt.getMatrixFromQuaternion(tgt_ort)
        tgt_R = np.array(tgt_R).reshape([3, 3])
        # Define the required transformation 
        trnsfm = SE3.Rt(tgt_R, tgt_pos)
        centroid_locs = trnsfm*SE3.Tx(pts[:, 0])*SE3.Ty(pts[:, 1])*SE3.Tz(pts[:, 2])
        # Return the translational part of the transformation
        return centroid_locs.t

    def detect_object_placement(self):
        """Use ray intersection to detect correct cube placement."""
        return utils.get_points_from_rays(
            self.sim, self._rays_from, self._rays_to, self._env_id
        )
