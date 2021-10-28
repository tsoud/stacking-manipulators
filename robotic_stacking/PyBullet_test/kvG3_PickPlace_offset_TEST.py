#!/usr/bin/env python

# --------------------------------------------------------------------------- #
'''
A planned pick and place script with a single arm using IK positioning.
Robot base is offset from the world origin and arm LCS is not aligned with
global CS. Movement and object locations are defined in the arm's local CS
instead.
'''
# --------------------------------------------------------------------------- #

import spatialmath.base as smb
from spatialmath import SE3, SO3
from spatialmath import UnitQuaternion as UQ

import os
import time
from collections import namedtuple
from pathlib import Path

import numpy as np
import pybullet as pbt
import pybullet_data

def deg_2_rad(angle):
    return np.pi * (angle / 180.)

def rad_2_deg(angle):
    return 180. * (angle / np.pi)

def unnamed_func(rotations:np.array, tol=5, unit='deg'):
    if unit == 'deg':
        tol = deg_2_rad(tol)
    rounded = np.rint(rotations)
    low, hi = (rounded - tol), (rounded + tol)
    if np.all(rotations > low) and np.all(rotations < hi):
        return rounded
    return rotations
    

# Give the physics simulation client a name.
# For example: sim1, sim2, ... and so on for multiple simulations.
# NOTE: This isn't necessary when running one simulation, but is used here
# clarity and making the code more applicable for general purposes.
sim1 = pbt.connect(pbt.GUI)
# Optional for loading objects from pybullet_data
pbt.setAdditionalSearchPath(pybullet_data.getDataPath())
# Set gravity constant
pbt.setGravity(gravX=0, gravY=0, gravZ=-9.8, physicsClientId=sim1)

# Make sure paths to environment objects are correct
resources_path = Path('../resources/').resolve()
robot_arm = os.path.join(resources_path, 'KinovaG3_7DOF_HandEgrip.urdf')
cube = os.path.join(resources_path, 'small_cube.urdf')
target_cube = os.path.join(resources_path, 'fake_cube.urdf')

# Load a plane and cube
floor_ID = pbt.loadURDF(
    "plane.urdf",
    [0, 0, 0],
    useFixedBase=True,
    physicsClientId=sim1
)
# Load a small cube
cube_ID = pbt.loadURDF(
    cube,
    basePosition=[-0.27, 0.583, 0.03],
    baseOrientation=pbt.getQuaternionFromAxisAngle(
        [0, 0, 1], (-np.pi / 7.8)),
    useMaximalCoordinates=True,
    physicsClientId=sim1
)
# Set a dummy target cube
target_location_ID = pbt.loadURDF(
    target_cube,
    basePosition=[0.5, -0.5, 0.03],
    baseOrientation=pbt.getQuaternionFromAxisAngle([0, 0, 1], -(np.pi / 3)),
    useFixedBase=True,
    physicsClientId=sim1
)

# Load the robot model.
random_base_pose = True
if random_base_pose:
    arm_base_pos = np.zeros((3,))
    arm_base_pos[0] = np.random.uniform(-0.1, 0.1)
    arm_base_pos[1] = np.random.uniform(-0.1, 0.1)
    angle = np.random.uniform(-(np.pi / 2), np.pi / 2)
    arm_base_ort = pbt.getQuaternionFromAxisAngle([0, 0, 1], angle)
    robot_arm_ID = pbt.loadURDF(
        robot_arm,
        useFixedBase=True,
        basePosition=arm_base_pos, 
        baseOrientation=arm_base_ort, 
        flags=(pbt.URDF_USE_SELF_COLLISION)
    )
else:
    robot_arm_ID = pbt.loadURDF(
        robot_arm,
        useFixedBase=True,
        flags=(pbt.URDF_USE_SELF_COLLISION)
    )

# Read information from the robot URDF description and get end link
n_joints = pbt.getNumJoints(robot_arm_ID)
joints_labeled = namedtuple('joints_labeled', ['ID', 'jointName', 'linkName'])
joint_labels = []
for jt_id in range(n_joints):
    jt_info = pbt.getJointInfo(robot_arm_ID, jt_id, sim1)
    jt_info = joints_labeled._make(list(jt_info[:2]) + [jt_info[12]])
    if jt_info.linkName.decode('UTF-8') == 'Robotiq_HandE_gripper':
        end_effector_id = jt_info.ID
    if jt_info.linkName.decode('UTF-8') == 'left_finger_assembly':
        grip_L = jt_info.ID
    if jt_info.linkName.decode('UTF-8') == 'right_finger_assembly':
        grip_R = jt_info.ID
    if jt_info.linkName.decode('UTF-8') == 'gripper_target':
        target_link_id = jt_info.ID
    joint_labels.append(jt_info)
# Joint between bracelet and gripper is fixed and can be ignored
arm_joint_ids = [j.ID for j in joint_labels[:end_effector_id]]
n_active_dofs = len(arm_joint_ids)
grip_ids = [grip_L, grip_R]
allJoints = arm_joint_ids + grip_ids
initial_states = pbt.getJointStates(robot_arm_ID, allJoints)
initial_states = [s[0] for s in initial_states]

# Joint damping coefficients to improve IK solution
# Grip fingers are directly actuated; use a high coefficient for them
jt_dmpg = [0.1] * 7 + [100.] * 2

# Enable force sensors for gripper
pbt.enableJointForceTorqueSensor(
    robot_arm_ID, jointIndex=grip_ids[0], enableSensor=True
)
pbt.enableJointForceTorqueSensor(
    robot_arm_ID, jointIndex=grip_ids[1], enableSensor=True
)


# --------------------------------------------------------------------------- #
# Functions for resolving coordinate transformations

def getRobotBasePose(robot=robot_arm_ID, AxisAngle=False):
    pose = pbt.getBasePositionAndOrientation(robot)
    posn, ornt = pose[0], pose[1]
    if AxisAngle:
        ornt = pbt.getAxisAngleFromQuaternion(ornt)
    return posn, ornt


base_pos, base_ort = getRobotBasePose()
lcs_transform = pbt.invertTransform(base_pos, base_ort)
wcs_transform = (base_pos, base_ort)

# Get a target world pose in robot's local cs
def wcs_2_lcs(target_pos, target_ort, transform=lcs_transform):
    return pbt.multiplyTransforms(
        transform[0], transform[1], target_pos, target_ort
        )

# Interpret a target position from robot lcs to wcs
def lcs_2_wcs(target_pos, target_ort, transform=wcs_transform):
    return pbt.multiplyTransforms(
        transform[0], transform[1], target_pos, target_ort
        )


def get_object_pose(object_ID, 
                    clearances=np.array([0., 0., 0.]), 
                    offsets=np.array([0., 0., 0.]), 
                    CS='local'):
    '''
    Returns the position and orientation of an object in the environment.
    ---
    CS: Coordinate system to use; 'local' returns values in robot's local CS,
        'world' is the global CS.
    clearances: x, y, z clearances to maintain from object to avoid collisions
    offsets: np array of x, y, z offset values to adjust the gripper fingers 
        with respect to the object being grasped.
    '''
    assert CS in ['local', 'world'], \
        f"`CS` parameter only takes 'local' or 'world' as arguments."
    object_posn, object_ornt = pbt.getBasePositionAndOrientation(object_ID)
    object_posn = np.array(object_posn) + clearances + offsets
    if CS == 'local':
        object_posn, object_ornt = wcs_2_lcs(object_posn, object_ornt)
    object_posn = np.asarray(object_posn, dtype=float)
    object_ornt = np.asarray(object_ornt, dtype=float)

    return object_posn, object_ornt


# --------------------------------------------------------------------------- #
# Functions to control pick and place actions
# --------------------------------------------------------------------------- #

def get_target_link_state(robot_arm_ID=robot_arm_ID,
                          target_link_id=target_link_id,
                          CS='local'):
    '''
    Returns the position and orientation of the robot's target link. The target 
    link is a "virtual" link that represents the part of the gripper that 
    should be aligned with a target object for proper gripping.
    ---
    CS: Coordinate system to use; 'local' returns values in robot's local CS,
        'world' is the global CS.
    '''
    assert CS in ['local', 'world'], \
        f"`CS` parameter only takes 'local' or 'world' as arguments."
    target_link_state = pbt.getLinkState(robot_arm_ID, target_link_id)[:2]
    # Return position and orientation
    position = target_link_state[0]
    orientation = target_link_state[1]
    if CS == 'local':
        position, orientation = wcs_2_lcs(position, orientation)

    return np.asarray(position), np.asarray(orientation)


def grip_finger_positions(grip_ids=grip_ids, bodyID=robot_arm_ID):
    states = pbt.getJointStates(bodyID, grip_ids)
    # return left and right grip finger positions
    return np.asarray([states[0][0], states[1][0]])


def get_arm_actuator_settings(target_position, target_orientation, 
                              use_robot_LCS=True, 
                              robot_arm_ID=robot_arm_ID,
                              target_id=target_link_id,
                              damping=jt_dmpg):
    ''' 
    Determine arm actuator angle settings from target pose using 
    inverse kinematics. The IK solver returns settings for all degrees 
    of freedom including gripper joints. Since gripper joints are actuated 
    independently, the last two values are removed from the result array.
    ---
    target_position: target position in robot local CS or world CS
    target_orientation: target orientation in robot local CS or world CS
    use_robot_LCS: (default == True); the target pose is given in the robot's 
        local coordinate system (LCS). In this case, a conversion is applied 
        to the target position and orientation to transform them into world 
        coordinates for the IK solver. 
        If false, the given target position and orientation are assumed to be 
        in world coordinates and directly given to the solver.
    '''
    if use_robot_LCS:
        target_position, target_orientation \
            = lcs_2_wcs(target_position, target_orientation)

    joint_angles = pbt.calculateInverseKinematics(
        bodyUniqueId=robot_arm_ID,
        endEffectorLinkIndex=target_id,
        targetPosition=target_position,
        targetOrientation=target_orientation, 
        jointDamping=damping
    )

    if target_orientation is None:
        joint_angles = pbt.calculateInverseKinematics(
            bodyUniqueId=robot_arm_ID,
            endEffectorLinkIndex=target_id,
            targetPosition=target_position, 
            jointDamping=damping
        )

    # return joint_angles
    return joint_angles[: -2]


def set_arm_actuators(joint_position_settings,
                      robot_arm_ID=robot_arm_ID,
                      joint_indices=arm_joint_ids):

    pbt.setJointMotorControlArray(
        bodyUniqueId=robot_arm_ID,
        jointIndices=joint_indices,
        controlMode=pbt.POSITION_CONTROL,
        targetPositions=joint_position_settings
    )


def set_grip_actuators(grip_position_settings, forces, 
                       robot_arm_ID=robot_arm_ID,
                       grip_indices=grip_ids):

    pbt.setJointMotorControlArray(
        bodyUniqueId=robot_arm_ID,
        jointIndices=grip_indices,
        controlMode=pbt.POSITION_CONTROL,
        targetPositions=grip_position_settings, 
        forces=forces
    )


def move_to_target(target_position, target_orientation,
                #    clearances=np.array([0., 0., 0.]), 
                #    offsets=np.array([0., 0., 0.]), 
                   num_steps=100,
                   robot_arm_ID=robot_arm_ID,
                   target_link_id=target_link_id):
    ''' 
    Move the end-effector to a target position using IK solver.
    ---
    target_position: x, y, z world coordinates of target (np array)
    target_orientation: world orientation quaternion x, y, z, w (np array)
    clearances: x, y, z clearances to maintain from target to avoid collisions
    offsets: np array of x, y, z offset values to adjust the gripper fingers 
        with respect to the object being grasped.
    num_steps: number of intermediate steps to smooth path
    '''
    done = False
    # target_position = target_position + clearances + offsets
    curr_posn, curr_ornt = get_target_link_state(
        robot_arm_ID, target_link_id, 
        )
    translation_stops = np.linspace(
        curr_posn, target_position,
        num=num_steps,
    )
    rotation_stops = np.linspace(
        curr_ornt, target_orientation,
        num=num_steps,
    )

    for step in range(1, num_steps):
        joint_angles = get_arm_actuator_settings(
            translation_stops[step], rotation_stops[step]
        )
        set_arm_actuators(joint_angles)
        pbt.stepSimulation(sim1)
        time.sleep(1. / 240.)

    done = True
    return done


# Calculate grip force for picking up objects
def calculate_grip_force(object_body_ID):
    ''' 
    Using the example in the Hand-E manual, we assume a real-world
    friction coefficient of 0.3 and safety factor of 2.4, which 
    gives a max object weight of 4.7 kg at the max force of 185 per 
    finger.
    ---
    object_body_ID: ID of the target object to grip
    '''
    object_mass = pbt.getDynamicsInfo(object_body_ID, -1)[0]
    return (object_mass / 4.7) * 185


# Get grip force for picking up objects
def get_grip_force(grip_ids=grip_ids, bodyID=robot_arm_ID):
    states = pbt.getJointStates(bodyID, grip_ids)
    # return left and right grip forces
    return np.asarray([states[0][3], states[1][3]])


def grasp_object(target_obj_ID, 
                 grip_increments=500, 
                 force_multiplier=1.0, 
                 bodyID=robot_arm_ID, 
                 grip_ids=grip_ids):
    '''
    Close the gripper fingers. Fingers keep moving until the grasping force 
    needed for securing the grip on the object is reached. 
    ---
    returns: `True` if enough force was generated to secure grip on item.
    ---
    grip_increments: n position increments for closing the grippers. Higher 
        values have smoother motion and reduce dynamic instabilities.
    force_multiplier: increases force to improve grip.
    '''    
    # Calculate required force to grasp the item properly
    required_force = calculate_grip_force(target_obj_ID)
    # print('\nreq force:', required_force, '\n')
    pos_limit = pbt.getJointInfo(robot_arm_ID, grip_ids[0])[9]
    force_limit = pbt.getJointInfo(robot_arm_ID, grip_ids[0])[10]
    # If a multiplier is used that exceeds the force limit, use force limit
    required_force = min(required_force * force_multiplier, force_limit)
    # Determine grip actuator increments
    grip_steps = np.linspace([0, 0], [pos_limit, pos_limit], grip_increments)
    
    for step in range(1, grip_increments):
        grip_force = get_grip_force()
        if np.all(grip_force <= required_force):
            set_grip_actuators(
                grip_steps[step], [force_limit, force_limit]
                )
            pbt.stepSimulation(sim1)
            time.sleep(1. / 240.)
        else:
            return True  # Success
        
    return False  # Could not generate enough force (e.g. object too heavy)


def release_object(increments=500, 
                   bodyID=robot_arm_ID, 
                   grip_ids=grip_ids):
    '''
    Open the gripper fingers. Returns the fingers to zero position. 
    ---
    returns: `True` when done.
    ---
    increments: used to calculate position step size. Higher values give 
        smoother and slower motion.
    '''
    pos_limit = pbt.getJointInfo(robot_arm_ID, grip_ids[0])[9]
    force_limit = pbt.getJointInfo(robot_arm_ID, grip_ids[0])[10]
    delta_pos = (pos_limit / increments) * (-1)
    crrnt_grip_pos = grip_finger_positions()
    L_steps = np.arange(crrnt_grip_pos[0], 0., delta_pos)
    R_steps = np.arange(crrnt_grip_pos[1], 0., delta_pos)
    n_steps = len(L_steps) if len(L_steps) == len(R_steps) \
        else min(len(L_steps), len(R_steps))
    grip_steps = np.stack((L_steps[:n_steps], R_steps[:n_steps]), axis=-1)

    for step in range(1, n_steps + 1):
        if step < n_steps:
            set_grip_actuators(
                grip_steps[step], [force_limit, force_limit]
                )
            pbt.stepSimulation(sim1)
            time.sleep(1. / 240.)
        elif step == n_steps:
            set_grip_actuators(
                [0., 0.], [force_limit, force_limit]
                )
            pbt.stepSimulation(sim1)
            time.sleep(1. / 240.)
            return True
        else:
            return False

# --------------------------------------------------------------------------- #
# GUI interface setup

# Key to reset arm to initial position
reset_key = ord('0')

# Add debugging lines to display target
loc1_id = pbt.addUserDebugLine(
    [0., 0., 0.125], [0.025, 0., 0.125],
    lineColorRGB=[1., 0.6, 0.2],
    lineWidth=3.5,
    parentObjectUniqueId=robot_arm_ID,
    parentLinkIndex=end_effector_id
)  # X - orange
loc2_id = pbt.addUserDebugLine(
    [0., 0, 0.125], [0., 0.025, 0.125],
    lineColorRGB=[1., 1., 0.2],
    lineWidth=3.5,
    parentObjectUniqueId=robot_arm_ID,
    parentLinkIndex=end_effector_id
)  # Y - yellow
loc3_id = pbt.addUserDebugLine(
    [0., 0., 0.125], [0., 0., 0.15],
    lineColorRGB=[1., 0.2, 1.],
    lineWidth=3.5,
    parentObjectUniqueId=robot_arm_ID,
    parentLinkIndex=end_effector_id
)  # Z - magenta

# Set GUI camera for better view
pbt.resetDebugVisualizerCamera(
    cameraDistance=1.4,
    cameraYaw=45,
    cameraPitch=-42,
    cameraTargetPosition=[0.25, 0, 0]
)

pbt.configureDebugVisualizer(pbt.COV_ENABLE_SHADOWS, 0)

# --------------------------------------------------------------------------- #

# Initialize arm actuator positions
pbt.setJointMotorControlArray(
    bodyUniqueId=robot_arm_ID,
    jointIndices=arm_joint_ids,
    controlMode=pbt.POSITION_CONTROL,
    targetPositions=[0.] * 7
)

# Use IK to return to "home" position when done
home_posn_w, home_ornt_w = pbt.getLinkState(robot_arm_ID, target_link_id)[:2]
# Get coordinates of target link in robot lcs
home_posn_loc, home_ornt_loc = wcs_2_lcs(home_posn_w, home_ornt_w)
home_ornt_w = pbt.getEulerFromQuaternion(home_ornt_w)
home_ornt_l_e = pbt.getEulerFromQuaternion(home_ornt_loc)
print('\nPose of target link in WCS:', home_posn_w, home_ornt_w, '\n')
print('\nPose of target link in LCS:', home_posn_loc, home_ornt_l_e, '\n')

clearances = np.array([0., 0., 0.1])
offsets = np.array([0., 0., 0.025])
# Position above cube
above_obj_posn, above_obj_ornt = get_object_pose(
    cube_ID, clearances=clearances,
    )
# Position at cube
at_obj_posn, at_obj_ornt = get_object_pose(cube_ID, offsets=offsets)
# Position above target
above_target_posn, above_target_ornt = get_object_pose(
    target_location_ID, clearances=clearances
    )
# Position at target
at_target_posn, at_target_ornt = get_object_pose(
    target_location_ID, offsets=offsets
    )
# Back at start position
home_posn_loc, home_ornt_loc = np.array(home_posn_loc), np.array(home_ornt_loc)

step_count = 0
start_target_reached = False
ready_to_grasp = False
item_grasped = False
item_picked = False
end_target_reached = False
ready_to_release = False
item_released = False
item_placed = False
returned_home = False

# --------------------------------------------------------------------------- #
# Run the pick and place sequence of actions
# --------------------------------------------------------------------------- #

while True:

    if step_count == 0:
        # Wait a little before moving
        time.sleep(5)
        step_count += 1
        start_action = True
        pbt.stepSimulation(sim1)

    if start_action:
        # Position robot end above cube
        start_target_reached = move_to_target(
            above_obj_posn, above_obj_ornt,
            num_steps=300,
            target_link_id=target_link_id
        )
        print('\nstart_target_reached:', start_target_reached, '\n')

    if start_target_reached:
        # Lower gripper to cube level
        ready_to_grasp = move_to_target(
            at_obj_posn, at_obj_ornt,
            num_steps=150, 
            target_link_id=target_link_id
            )
        print('\nready_to_grasp:', ready_to_grasp, '\n')

    if ready_to_grasp:
        # Grip the cube
        item_grasped = grasp_object(cube_ID, force_multiplier=1.5)
        print('\nitem_grasped:', item_grasped, '\n')

    if item_grasped:
        # Move back to the position above the cube
        item_picked = move_to_target(
            above_obj_posn, above_obj_ornt,
            num_steps=150,
            target_link_id=target_link_id
            )
        print('\nitem_picked:', item_picked, '\n')

    if item_picked:
        time.sleep(1)
        # Move to target location and position end slightly above
        end_target_reached = move_to_target(
            above_target_posn, above_target_ornt, 
            num_steps=500, 
            target_link_id=target_link_id
            )
        print('\nend_target_reached:', end_target_reached, '\n')

    if end_target_reached:
        time.sleep(1)
        # Lower gripper to place cube
        ready_to_release = move_to_target(
            at_target_posn, at_target_ornt, 
            # small clearance to avoid floor collision
            num_steps=150, 
            target_link_id=target_link_id
            )
        print('\nready_to_release:', ready_to_release, '\n')
    
    if ready_to_release:
        # Release item
        item_released = release_object()
        print('\nitem_released:', item_released, '\n')

    if item_released:
        # Move gripper away
        item_placed = move_to_target(
            above_target_posn, above_target_ornt, 
            num_steps=150, 
            target_link_id=target_link_id
            )
        print('\nitem_placed:', item_placed, '\n')

    if item_placed:
        # Return to home position
        time.sleep(1)
        returned_home = move_to_target(
            home_posn_loc, home_ornt_loc, 
            num_steps=300, 
            target_link_id=target_link_id
            )
        print('\nreturned_home:', returned_home, '\n')

    if returned_home:
        # Automatically shut down after 2 min if GUI window isn't closed
        for _ in range(120 * 240):
            pbt.stepSimulation(sim1)
            time.sleep(1. / 240.)
        pbt.disconnect()
