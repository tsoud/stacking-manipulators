import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pybullet as pbt

from robotic_stacking import assets, utils

# ----------------------------------------------------------------------------
# Define and load a robot model.
# ----------------------------------------------------------------------------

class robot(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @property
    @abstractmethod
    def joints(self):
        pass

    @property
    @abstractmethod
    def links(self):
        pass

    @abstractmethod
    def get_description(self):
        pass

# ----------------------------------------------------------------------------

class urdf_robot(robot):
    """ 
    Set robot attributes from a URDF file.

    keyword args:
    -------------
    robot_model: Select the robot model to use from the models 
        available in `robotic_stacking.assets`.
    """
    def __init__(self, robot_model:str):
        self.model = robot_model
        self.__urdf_path = assets.find_urdf_objects().get(robot_model)
        self.__joints = {}
        self.__links = {}
        # self.__motor_force_limits = {}
        # Parse the URDF to map joints and links to PyBullet int index
        jt_idx, lk_idx = 0, 0
        tree = ET.parse(self.__urdf_path)
        joint_info = namedtuple(
            'joint_info', 'PyBullet_idx, type, max_force, max_velocity'
        )
        self.__urdf_root = tree.getroot()
        for child in self.__urdf_root:
            if child.tag == 'joint':
                name = child.attrib.get('name')
                jt_type = child.attrib.get('type')
                # Parse actuator motor force and velocity limits
                # This is can be helpful for setting joint positions
                limit = child.find('./limit')
                if limit is None:
                    max_force = None
                    max_velocity = None
                else:
                    max_force = int(limit.attrib.get('effort'))
                    max_velocity = float(limit.attrib.get('velocity'))
                self.__joints[name] = joint_info(
                    jt_idx, jt_type, max_force, max_velocity
                )
                jt_idx += 1
            if (child.tag == 'link' and 
                not child.attrib.get('name') == 'base_link'):
                self.__links[child.attrib.get('name')] = lk_idx
                lk_idx += 1

    @property
    def URDF(self):
        return self.__urdf_path
    
    @property
    def joints(self):
        return self.__joints

    @property
    def links(self):
        return self.__links

    @property
    def available_DOF(self):
        """Degrees of freedom of arm excluding gripper DOF."""
        pass

    @property
    def grip_ids(self):
        pass

    @property
    def target_link_id(self):
        """A link for aligning the gripper with a target."""
        pass

    @property
    def active_joints(self):
        """All the robot's movable joints, excludes fixed joints."""
        pass

    def get_description(self):
        """Prints a description of the robot structure"""
        root = self.__urdf_root
        print(f"\nRobot structure as [parent_link]-(joint)-[child_link]:"
                + "\n" + ("-" * 79))
        for jt, pt, ch in zip(
            root.iter('joint'), root.iter('parent'), root.iter('child')
            ):
            print(
                f"[Parent:{pt.attrib.get('link')}]-->"
                + f"({jt.attrib.get('name')}, type:{jt.attrib.get('type')})"
                + f"-->[Child:{ch.attrib.get('link')}]"
            )
        print(("-" * 79) + "\n")

# ----------------------------------------------------------------------------
    
class kinovaG3_7D_HandE(urdf_robot):
    """ 
    Define a Kinova Gen3 7-DOF robotic arm.

    keyword args:
    -------------
    robot_model: Select the robot model to use from the models 
        available in `robotic_stacking.assets`.
    """
    def __init__(self):
        super().__init__(robot_model='KinovaG3_7DOF_HandEgrip')
        self._arm_actuators = [
            self.joints.get(j).PyBullet_idx for j in self.joints.keys() 
            if j.startswith('Actuator')
        ]
        self._motor_force_limits = np.array([
            j.max_force for j in self.joints.values() 
            if j.PyBullet_idx in self.active_joints
        ])
        self._motor_velocity_limits = np.array([
            j.max_velocity for j in self.joints.values() 
            if j.PyBullet_idx in self.active_joints
        ])

    @property
    def arm_actuators(self):
        return self._arm_actuators

    @property
    def available_DOF(self):
        """Degrees of freedom of arm excluding gripper DOF."""
        return len(self.arm_actuators)

    @property
    def grip_ids(self):
        return [
            self.links.get('left_finger_assembly'), 
            self.links.get('right_finger_assembly')
        ]

    @property
    def target_link_id(self):
        """A link for aligning the gripper with a target."""
        return self.links.get('gripper_target')

    @property
    def active_joints(self):
        """All the robot's movable joints, excludes fixed joints."""
        return self.arm_actuators + self.grip_ids

    @property
    def motor_force_limits(self):
        """Max force limits for actuator motors."""
        return self._motor_force_limits

    @property
    def motor_velocity_limits(self):
        """Max force limits for actuator motors."""
        return self._motor_velocity_limits

# ----------------------------------------------------------------------------
# Define the robot controllers for interacting with PyBullet.
# ----------------------------------------------------------------------------

class robotic_arm_controller:
    """
    Functions to control a robotic arm in a PyBullet simulation.

    NOTE: methods defined with `pass` are placeholders to be over-
    written for specific robot models.

    keyword args:
    -------------
    simulation: the PyBullet simulation with which the robot interacts.
    robot: a `robot` class to add to the simulation.
    base_position: robot base position (x, y, z) in world coordinates.
    base_orientation: robot base orientation (x, y, z, w) quaternion 
        in world coordinates.
    end_effector: the robot link that will do the work. This can be a 
        real link like the body of a gripper or a virtual link 
        representing a target that moves with the robot.
    fixed_base: use a stationary robot base.
    use_robot_lcs: use the robot's local coordinate system to 
        determine locations and orientations, select poses, and control 
        movements. Defaults to `True` to ensure the agent uses its 
        own representation of the environment to interpret 
        observations and select actions.
    use_grip_force_sensors: activate force sensors in the gripper 
        fingers.
    alt_initial_actuator_posns: set the arm to an alternate starting 
        configuration by changing the initial position of the arm 
        actuators. Otherwise, all actuators are set to a zero 
        position.
    joint_damping: list of joint damping factors for improving the
        IK solution. Size of the list should match the number of 
        active DOF. If `None`, a default list of small values is used.
    """
    def __init__(self, 
                 simulation, 
                 robot, 
                 base_position:Union[Tuple, List], 
                 base_orientation:Union[Tuple, List], 
                 end_effector:int, 
                 fixed_base:bool=True, 
                 use_robot_lcs:bool=True, 
                 alt_initial_actuator_posns:Optional[List]=None, 
                 joint_damping:Optional[List]=None):
        
        self._sim = simulation
        self._robot = robot
        self._robot_id = self._sim.loadURDF(
            self._robot.URDF, 
            basePosition=base_position, 
            baseOrientation=base_orientation, 
            useFixedBase=fixed_base, 
            flags=pbt.URDF_USE_SELF_COLLISION
        )
        self._base_pos, self._base_ort = utils.getRobotBasePose(
            self._sim, self._robot_id, False
        )
        self._end_effector = end_effector
        # Calculate transformations from world to local and vice versa
        self._use_lcs = use_robot_lcs
        self._wcs_2_lcs = pbt.invertTransform(self._base_pos, self._base_ort)
        self._lcs_2_wcs = (self._base_pos, self._base_ort)
        # break out WCS to LCS translation and rotation
        self._wcs2lcs_tr = np.array(self._wcs_2_lcs[0])
        self._wcs2lcs_rot = np.array(self._wcs_2_lcs[1])
        self._lcs2wcs_tr = np.array(self._lcs_2_wcs[0])        
        self._lcs2wcs_rot = np.array(self._lcs_2_wcs[1])
        # Initialize robot actuators
        if alt_initial_actuator_posns is None:
            self._initial_jt_config = [0.]*self._robot.available_DOF
            
        else:
            self._initial_jt_config = alt_initial_actuator_posns
        self._sim.setJointMotorControlArray(
            bodyUniqueId=self._robot_id,
            jointIndices=self._robot.arm_actuators,
            controlMode=pbt.POSITION_CONTROL,
            targetPositions=self._initial_jt_config
        )
        if joint_damping is None:
            self._jt_dmpg = [1e-3]*self._robot.active_joints
        else:
            self._jt_dmpg = joint_damping
        self.__robot_info = namedtuple(
            'robot_info', 'model, base_position, base_orientation'
        )

    def __repr__(self):
        robot_info = self.__robot_info(
            self._robot.model, self._base_pos, self._base_ort
        )
        return str(robot_info)
    
    @property
    def robot_id(self):
        """Returns robot's unique ID in the simulation environment."""
        return self._robot_id
    
    @property
    def robot_base_pose(self):
        """Returns the robot's base position and orientation."""
        return self._base_pos, self._base_ort

    @property
    def using_local_CS(self):
        """Returns `True` if robot is using local coordinates."""
        return self._use_lcs
        
    def wcs_2_lcs(self, world_pos, world_ort):
        """Transform world coordinates to local robot coordinates."""
        return self._sim.multiplyTransforms(
            self._wcs_2_lcs[0], self._wcs_2_lcs[1], world_pos, world_ort
        )
    
    def lcs_2_wcs(self, local_pos, local_ort):
        """Transform local robot coordinates to world coordinates."""
        return self._sim.multiplyTransforms(
            self._lcs_2_wcs[0], self._lcs_2_wcs[1], local_pos, local_ort
        )

    def pts_2_lcs(self, points_wcs:np.array):
        """
        Convert an array of points from world CS to local.
        
        When tracking only arrays of points, this method is quicker 
        than calling `wcs_2_lcs()` multiple times.

        keyword args:
        ------------
        points_wcs: array of points in world coordinates.

        returns:
        -------
        A Numpy array of point coordinates in the robot LCS. If input 
        is a three-dimensional array (representing point arrays from 
        multiple objects), output array is also a 3D array.
        """
        n_dim = points_wcs.ndim
        if not n_dim in (2, 3):
            raise ValueError('Input array must have either 2 or 3 dimensions')
        if n_dim == 3:
            n_objs = points_wcs.shape[0]
            wcs2lcs_tr = np.tile(self._wcs2lcs_tr, (points_wcs.shape[1], 1))
            wcs2lcs_tr = np.stack([wcs2lcs_tr]*n_objs, axis=0)
        else:
            wcs2lcs_tr = np.tile(self._wcs2lcs_tr, (points_wcs.shape[0], 1))

        return wcs2lcs_tr + points_wcs

    def get_end_eff_pose(self):
        """
        Returns position and orientation of the robot's end-effector.
        
        If `use_robot_lcs` is specified in the constructor, pose is 
        calculated in the robot's local CS.
        """
        link_state = self._sim.getLinkState(
            self._robot_id, linkIndex=self._end_effector
        )[:2]
        # Return position and orientation
        position = link_state[0]
        orientation = link_state[1]
        if self._use_lcs:
            position, orientation = self.wcs_2_lcs(position, orientation)
        return np.array(position), np.array(orientation)

    def grip_finger_positions(self) -> np.array:
        """Returns current actuator positions of the grip fingers."""
        states = self._sim.getJointStates(self._robot_id, self._robot.grip_ids)
        # return left and right grip finger positions
        return np.array([states[0][0], states[1][0]])
    
    def detect_grip_contact(self, object_ID:int) -> np.array:
        """ 
        Detect if gripper contacts an object in the environment.

        returns:
        -------
        A numpy boolean array with each True/False element indicating 
        whether a gripper link (e.g. a finger) is in contact with the 
        object given by `object_ID`.
        """
        pass

    def get_grip_force(self):
        """Returns actuator force applied to each gripper."""
        pass

    def detect_self_collision(self) -> bool:
        """
        Detects if any link in the robot arm collides with another.

        returns:
        -------
        `True` if self-collision is detected.
        """
        self_collision = self._sim.getContactPoints(
            bodyA=self._robot_id, bodyB=self._robot_id
        )
        return True if self_collision else False

    def get_link_poses(self, links:Optional[
                                Union[List[int], Tuple[int], np.array]
                                ]=None) -> Tuple[np.array, np.array]:
        """
        Get position and orientation of center of mass of arm links.
        
        keyword args:
        ------------
        links: array of specific links, represented by their integer 
            simulation IDs. Default is all the robot's links.
        """
        links = self._robot.arm_actuators if links is None else links
        link_poses = self._sim.getLinkStates(self._robot_id, links)
        if self._use_lcs:
            link_poses = [self.wcs_2_lcs(s[0], s[1]) for s in link_poses]
        else:
            link_poses = [(s[0], s[1]) for s in link_poses]
        link_poses = list(zip(*link_poses))
        return np.array(link_poses[0]), np.array(link_poses[1])

    def get_arm_actuator_settings(self, 
                                  target_position:np.array, 
                                  target_orientation:np.array) -> np.array:
        """ 
        Get the arm actuator angles to reach a desired pose.

        This function uses PyBullet's built-in IK solver to determine 
        the joint settings.

        keyword args:
        ------------
        target_position, target_orientation: the new position and 
            orientation desired. If the robot is using its local 
            coordinate system (the default), these should be given in 
            local coordinates.
        """
        # PyBullet IK solver uses world coordinates, so transform to wcs when 
        # `use_robot_lcs` is `True`.
        if self._use_lcs:
            target_position, target_orientation = self.lcs_2_wcs(
                target_position, target_orientation
            )
        # NOTE: the IK solver has to use ALL non-fixed joints in the robot's 
        # kinematic chain, including gripper joints. High damping values for 
        # the gripper joints can greatly improve the results because their 
        # impact on the solution becomes practically negligible.
        joint_angles = self._sim.calculateInverseKinematics(
            bodyUniqueId=self._robot_id,
            endEffectorLinkIndex=self._end_effector,
            targetPosition=target_position,
            targetOrientation=target_orientation, 
            jointDamping=self._jt_dmpg, 
            maxNumIterations=25,
            residualThreshold=1e-6
        )
        # return the calculated actuator angles
        return np.array(joint_angles)
    
    def get_actuator_sequence(self, 
                              dx_dy_dz:np.array, 
                              dRx_dRy_dRz:np.array,
                              delta_grip:float, 
                              num_steps:int):
        """
        Returns a sequence of settings to apply at each action step.

        This function translates the selected actions into a sequence 
        of position settings for the robot's actuators.

        keyword args:
        ------------
        dx_dy_dz: relative end-effector translation increments 
            in x, y, z directions. Uses the robot local CS if 
            `use_robot_lcs` is `True`.
        dRx_dRy_dRz: relative end-effector rotation increments about 
            x, y, z, axes. Uses the robot local CS if 
            `use_robot_lcs` is `True`.
        delta_grip: move gripper fingers by this amount. Positive 
            values close the grip fingers, negative values open.
        num_steps: number of simulation steps over which to increment 
            the settings. This should equal the number of simulation 
            substeps in a global transition step.
        """
        pass

    def set_actuators(self, actuators, positions, 
                      position_gains:Optional[List]=None):
        """Set actuators to move the arm and gripper."""
        pass

    def reset(self):
        """
        Reset all the robot's actuators to their initial positions.
        """
        pass
    
    def delete(self):
        """Remove the robot from the environment."""
        self._sim.removeBody(self._robot_id)


# ----------------------------------------------------------------------------

class kvG3_7_HdE_control(robotic_arm_controller):
    """ 
    Specific controller for the Kinova Gen3 7-DOF arm with Hand-E grip.

    keyword args:
    -------------
    simulation: the PyBullet simulation in which the robot exists.
    base_position: robot base position (x, y, z) in world coordinates.
    base_orientation: robot base orientation (x, y, z, w) quaternion 
        in world coordinates.
    fixed_base: use a stationary robot base.
    use_robot_lcs: use the robot's local coordinate system to 
        determine locations and orientations, select poses, and control 
        movements. Defaults to `True` to ensure the agent uses its 
        own representation of the environment to interpret 
        observations and select actions.
    use_grip_force_sensors: activate force sensors in the gripper 
        fingers.
    alt_initial_actuator_posns: set the arm to an alternate starting 
        configuration by changing the initial position of the arm 
        actuators. Otherwise, all actuators are set to a zero 
        position.
    """
    def __init__(self, 
                 simulation, 
                 base_position:Union[Tuple, List, np.array]=(0., 0., 0.), 
                 base_orientation:Union[Tuple, List, np.array]=(0., 0., 0., 1.), 
                 fixed_base:bool=True, 
                 use_robot_lcs:bool=True, 
                 alt_initial_actuator_posns:Optional[List]=None):

        super().__init__(
            simulation=simulation, 
            robot=kinovaG3_7D_HandE(),  
            base_position=base_position, 
            base_orientation=base_orientation, 
            end_effector=10, 
            joint_damping=[0.1]*7 + [100.]*2, 
            fixed_base=fixed_base, 
            use_robot_lcs=use_robot_lcs, 
            alt_initial_actuator_posns=alt_initial_actuator_posns 
        )
        # default position gains for actuator setting
        self._actuator_position_gains = [1.]*len(self._robot.active_joints)

    def get_actuator_sequence(self, 
                              dx_dy_dz:np.array=np.array([0., 0., 0.]), 
                              dRx_dRy_dRz:np.array=np.array([0., 0., 0.]), 
                              delta_grip:float=0., 
                              num_steps:int=100):
        """
        Returns a sequence of settings to apply at each action step.

        This function translates the actions selected in a transition 
        step into a sequence of position settings for the robot's 
        actuators.

        keyword args:
        ------------
        dx_dy_dz: relative end-effector translation increments 
            in x, y, z directions. Uses the robot local CS if 
            `use_robot_lcs` is `True`.
        dRx_dRy_dRz: relative end-effector rotation increments about 
            x, y, z, axes. Uses the robot local CS if 
            `use_robot_lcs` is `True`.
        delta_grip: move gripper fingers by this amount. Positive 
            values close the grip fingers, negative values open.
        num_steps: number of simulation steps over which to increment 
            the settings. This should equal the number of simulation 
            substeps in a global transition step.

        returns:
        -------
        A sequence of actuator positions as a numpy array with dims 
        (num_steps x n_actuators) and the intended goal position and 
        orientation based on the inputs.
        """
        # get current pose
        curr_pos, curr_ort = self.get_end_eff_pose()
        # determine new pose
        goal_position = curr_pos + dx_dy_dz
        goal_orientation = utils.increment_RxRyRz(curr_ort, dRx_dRy_dRz)
        new_grip_pos = (
            np.array([delta_grip, delta_grip]) + self.grip_finger_positions()
        )
        # calculate incremental stops
        translation_stops = np.linspace(
            curr_pos, goal_position,
            num=num_steps,
        )
        rotation_stops = utils.q_interp(
            curr_ort, goal_orientation, num_steps=num_steps
        )
        # get current gripper finger positions and calculate increments
        curr_grip_pos = self.grip_finger_positions()
        delta_grip = np.array([delta_grip, delta_grip])
        grip_stops = utils.calculate_grip_stops(
            curr_grip_pos, delta_grip, n_steps=num_steps
        )
        # Create an empty array for the sequence of actuator settings
        sequence = np.empty(
            (
                num_steps, 
                len(self._robot.arm_actuators) + len(self._robot.grip_ids)
            ), dtype=float
        )
        # Assign actuator settings
        # NOTE: the first row from `get_arm_actuator_settings()` is the 
        # current position and orientation setting so it is skipped.
        for step in range(1, num_steps):
            # Assign arm motor settings, skip gripper actuators
            sequence[step - 1, : -len(self._robot.grip_ids)] = (
                self.get_arm_actuator_settings(
                    translation_stops[step], rotation_stops[step]
                )[: -len(self._robot.grip_ids)]
            )
            # Assign gripper actuator settings directly
            sequence[step-1, -len(self._robot.grip_ids):] = grip_stops[step]
        # Repeat the last row as the last assignment (helps stabilize forces)
        sequence[-1] = sequence[-2]

        return sequence, goal_position, goal_orientation

    def set_actuators(self, positions:Union[List, Tuple, np.array], 
                      joints:Optional[List]=None,  
                      position_gains:Optional[List]=None):
        """
        Set actuators to move the arm and gripper.
        
        positions: (list or array-like) New positions to set actuators.
        joints: (list or array-like) A list of the actuators to set. 
            Defaults to all active joints if `None`. Length of this 
            list should match the number of positions given.
        position_gains: (list or array-like) Optional list of position 
            gains to pass to PyBullet's `setJointMotorControlArray()` 
            function. Length of the array should match the number of 
            positions given. By default, all values are set to 1. 
            Other values may lead to inaccurate or unexpected 
            behavior, so it is best to leave this as-is unless the 
            user is modeling a specific PD control and has strong 
            knowledge of its behavior.
        """
        if joints is None:
            joints = self._robot.active_joints
            max_forces = self._robot.motor_force_limits
        else:
            joints = joints
            max_forces = self._robot.motor_force_limits[
                np.in1d(self._robot.active_joints, joints)
            ]
        if position_gains is None:
            position_gains = self._actuator_position_gains
        else:
            position_gains = position_gains

        pbt.setJointMotorControlArray(
            bodyUniqueId=self._robot_id,
            jointIndices=joints,
            controlMode=pbt.POSITION_CONTROL,
            targetPositions=positions, 
            forces=max_forces, 
            positionGains=position_gains
        )
        
    def detect_grip_contact(self, object_ID:int) -> np.array:
        """ 
        Detect if gripper fingers contact an object in the environment.

        returns:
        -------
        A numpy boolean array with each True/False element indicating 
        whether each finger is in contact with the object given by 
        `object_ID`.
        """
        left, right = self._robot.grip_ids[0], self._robot.grip_ids[1]
        left_contact = self._sim.getContactPoints(
            bodyA=self._robot_id, bodyB=object_ID, linkIndexA=left
        )
        right_contact = self._sim.getContactPoints(
            bodyA=self._robot_id, bodyB=object_ID, linkIndexA=right
        )
        left_contact = True if len(left_contact) > 0 else False
        right_contact = True if len(right_contact) > 0 else False

        return np.array([left_contact, right_contact], dtype=bool)

    def get_grip_force(self):
        """Returns actuator force applied to each gripper."""
        states = self._sim.getJointStates(
            self._robot_id, self._robot.grip_ids
        )
        # return left and right grip forces
        return np.array([states[0][3], states[1][3]])
    
