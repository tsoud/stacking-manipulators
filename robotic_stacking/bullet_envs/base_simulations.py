import time
from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
from typing import List, Optional, Tuple, Union

import pybullet_data
from PIL import Image

from robotic_stacking import robot
from robotic_stacking.pybullet_connections import single_connection

# ----------------------------------------------------------------------------
# Set up the base PyBullet simulation environment.
# ----------------------------------------------------------------------------


class simulation(ABC):

    @abstractmethod
    def __init__(self, connection_type, use_GUI):
        pass

    @property
    @abstractmethod
    def sim(self):
        pass

    @property
    @abstractmethod
    def sim_id(self):
        pass

    @abstractmethod
    def add_robot(self):
        pass

    @abstractmethod
    def add_env_object(self):
        pass

    @abstractmethod
    def reset_robot(self):
        pass

    @abstractmethod
    def reset_env_object(self):
        pass

    @abstractmethod
    def simulation_step(self):
        pass

    @abstractmethod
    def close(self):
        pass


# ----------------------------------------------------------------------------

class single_agent_env(simulation):
    """
    Set up a PyBullet simulation environment with a single robot arm.

    The simulation uses only `single_connection()` to connect to the 
    physics server.

    keyword args:
    ------------
    use_GUI: run the simulation in a graphics window.
    gravity: gravity constant m/sec^2 in world X, Y, Z directions. 
        Z is perpendicular to ground. Defaults to (0., 0., -9.8).
    """
    def __init__(self, use_GUI=False, gravity=(0., 0., -9.8)):

        self._use_GUI = 'GUI' if use_GUI else 'direct'
        self._connection = single_connection(self._use_GUI)
        self._connection.connect()
        self._sim = self._connection.client
        self._sim_id = self._connection.client_id
        # add path to pybullet data
        self._sim.setAdditionalSearchPath(pybullet_data.getDataPath())
        # define gravity
        self._sim.setGravity(*gravity)
        # add a floor to the simulation
        self._floor_surface = self._sim.loadURDF("plane.urdf")
        # track robot information
        self._robot = None
        self._robot_id = None
        # keep track of initial robot parameters for resetting
        self.__robot_init_params = {}
        self._env_objs = defaultdict(namedtuple)

    @staticmethod
    def get_dict_from_repr(repr_text):
        """
        Extract a dict from a robot or other object's __repr__.

        Assumes the __repr__ represents a `namedtuple` type, which is 
        the case for any `robot_controller` class.
        """
        text = repr_text.partition('(')
        text = text[-1].replace('=array(', '=tuple(')
        return eval('dict(' + text)

    @property
    def sim(self):
        return self._sim

    @property
    def sim_id(self):
        return self._sim_id

    @property
    def arm_control(self):
        return self._robot

    @property
    def env_objects(self):
        return self._env_objs

    @property
    def env_object_ids(self):
        return self._env_objs.keys()

    def add_robot(self, 
                  robot_controller=robot.kvG3_7_HdE_control, 
                  base_position:Union[Tuple, List]=(0., 0., 0.), 
                  base_orientation:Union[Tuple, List]=(0., 0., 0., 1.), 
                  controller_kwargs:Optional[dict]=None):
        """
        Add a robot to the simulation.

        keyword args:
        ------------
        robot_controller: the `robot_controller` class to use.
        base_position, base_orientation: robot base pose in 
            world coordinates.
        controller_kwargs: dict of additional keyword args to pass 
            to the `robot_controller` class.

        Updates `self._robot` and `self._robot_id` attributes with 
        the controller class and robot environment ID.
        """
        if self._robot:
            return 'A robot already exists in this environment.'
        if controller_kwargs is None:
            self._robot = robot_controller(
                self._sim, 
                base_position=base_position, base_orientation=base_orientation 
            )
        else:
            self._robot = robot_controller(
                self._sim, 
                base_position=base_position, base_orientation=base_orientation, 
                **controller_kwargs
            )
        self._robot_id = self._robot.robot_id
        self.__robot_init_params = self.get_dict_from_repr(self._robot.__repr__())
        self.__robot_init_params['init_position'] = base_position
        self.__robot_init_params['init_orientation'] = base_orientation
        self.__robot_init_params['controller_kwargs'] = controller_kwargs

    def reset_robot(self, 
                    new_position:Optional=None, 
                    new_orientation:Optional=None):
        """
        Reset the robot in the current environment.

        This function simulates a robot reset by overwriting the 
        existing robot instance with a new one. If a new position 
        and/or orientation are given, the robot is instantiated with 
        the new pose, otherwise it uses the previous pose.
        """
        controller = self._robot.__class__
        controller_kwargs = self.__robot_init_params.get('controller_kwargs')
        if new_position is None:
            base_pos = self.__robot_init_params.get('init_position')
        else:
            base_pos = new_position
        if new_orientation is None:
            base_ort = self.__robot_init_params.get('init_orientation')
        else:
            base_ort = new_orientation
        # remove existing robot and initialize attributes
        self._robot.delete()
        self._robot, self._robot_id, self.__robot_init_params = None, None, None
        # instantiate a new robot
        self.add_robot(
            robot_controller=controller, 
            base_position=base_pos, 
            base_orientation=base_ort, 
            controller_kwargs=controller_kwargs)

    def add_env_object(self, env_object, 
                       position, orientation, 
                       object_kwargs:Optional[dict]=None):
        """
        Add a non-robot object to the simulation.

        keyword args:
        ------------
        env_object: an object class to use. The class should take the 
            simulation and a position and orientation as args to its 
            constructor.
        position, orientation: object pose in world coordinates.
        object_kwargs: dict of additional keyword args to pass 
            to the object class.
        """
        if object_kwargs is None:
            new_env_obj = env_object(
                self._sim, position=position, orientation=orientation,  
            )
        else:
            new_env_obj = env_object(
                self._sim, position=position, orientation=orientation, 
                **object_kwargs
            )
        self._env_objs[new_env_obj._env_id] = new_env_obj
        return new_env_obj._env_id

    def reset_env_object(self, env_obj_id, 
                         new_position:Optional=None, 
                         new_orientation:Optional=None):
        """
        Reset an environment object.

        The objects forces and all related dynamics are removed. If a 
        new position and/or orientation are given, the object assumes 
        the new pose, otherwise it is reset to its initial pose when 
        first added to the simulation.
        """
        env_obj = self._env_objs.get(env_obj_id)
        env_obj.reset(new_position, new_orientation)

    def remove_env_object(self, env_object_id):
        """
        Remove an environment object from the simulation.

        keyword args:
        ------------
        env_object: an object class to use. The class should take the 
            simulation and a position and orientation as args to its 
            constructor.
        position, orientation: object pose in world coordinates.
        object_kwargs: dict of additional keyword args to pass 
            to the object class.
        """
        if env_object_id not in self._env_objs.keys():
            raise KeyError(f'No object with ID:{env_object_id} found.')
        env_object = self._env_objs.get(env_object_id)
        # check if object was deleted (certain items cannot be deleted)
        deleted = env_object.delete()
        if deleted:
            _ = self._env_objs.pop(env_object_id)

    def simulation_step(self, sleep:Optional[float]=None):
        """
        Move the physics simulation forward exactly one time step.

        NOTE: This is not the same as the transition step used 
        by an RL agent. A single transition step can have one or more 
        physics simulation substeps.
        """
        self._sim.stepSimulation()
        if (sleep and self._use_GUI):
            time.sleep(sleep)

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
        view_matrix = self._sim.computeViewMatrixFromYawPitchRoll(*camera_view)      
        if projection_matrix is None:
            proj_matrix = self._sim.computeProjectionMatrixFOV(
                60, (img_size[0]/img_size[1]), 0.1, 100
            )
        else:
            proj_matrix = projection_matrix
        img_data = self._sim.getCameraImage(
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
        self._connection.close()

# ----------------------------------------------------------------------------


class multi_agent_env(simulation):
    """
    Set up a PyBullet simulation environment with multiple robot arms.

    The simulation can connect to the physics server with either
    `single_connection()` or `multiclient_server()` connection types.

    keyword args:
    ------------
    use_GUI: run the simulation in a graphics window.
    gravity: gravity constant m/sec^2 in world X, Y, Z directions. 
        Z is perpendicular to ground. Defaults to (0., 0., -9.8).
    """
    pass

# ----------------------------------------------------------------------------
