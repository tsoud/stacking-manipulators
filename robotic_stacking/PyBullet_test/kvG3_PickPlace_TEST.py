#!/usr/bin/env python

# --------------------------------------------------------------------------- #
'''
A planned pick and place script with a single arm that purely uses IK. 
This script can be used for testing.
'''
# --------------------------------------------------------------------------- #

import os
import time
from collections import namedtuple
from pathlib import Path

import numpy as np
import pybullet as pbt
import pybullet_data

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
# Functions to control pick and place actions 
# --------------------------------------------------------------------------- #


def get_target_link_state(robot_arm_ID=robot_arm_ID,
                          target_link_id=target_link_id):
    target_link_state = pbt.getLinkState(robot_arm_ID, target_link_id)[:2]
    # Return position and orientation
    position = target_link_state[0]
    orientation = target_link_state[1]

    return np.asarray(position), np.asarray(orientation)


def grip_finger_positions(grip_ids=grip_ids, bodyID=robot_arm_ID):
    states = pbt.getJointStates(bodyID, grip_ids)
    # return left and right grip finger positions
    return np.asarray([states[0][0], states[1][0]])


def get_arm_actuator_settings(target_position, target_orientation,
                              robot_arm_ID=robot_arm_ID,
                              target_id=target_link_id,
                              damping=jt_dmpg):
    ''' 
    Determine arm actuator angle settings from target pose using 
    inverse kinematics. The IK solver returns settings for all degrees 
    of freedom including gripper joints. Since gripper joints are actuated 
    independently, the last two values are removed from the result array.
    '''
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


def get_object_pose(object_ID):
    object_pose = pbt.getBasePositionAndOrientation(object_ID)
    object_position = np.asarray(object_pose[0], dtype=float)
    object_orientation = np.asarray(object_pose[1], dtype=float)
    return object_position, object_orientation


def move_to_target(target_position, target_orientation,
                   clearances=np.array([0., 0., 0.]), 
                   offsets=np.array([0., 0., 0.]), 
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
    target_position = target_position + clearances + offsets
    curr_posn, curr_ornt = get_target_link_state(robot_arm_ID, target_link_id)
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
        if np.any(grip_force < required_force):
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
home_pose = pbt.getLinkState(robot_arm_ID, target_link_id)

clearances = np.array([0., 0., 0.1])
offsets = np.array([0., 0., 0.02])
start_target_posn, start_target_ornt = get_object_pose(cube_ID)
end_target_posn, end_target_ornt = get_object_pose(target_location_ID)
home_posn, home_ornt = np.array(home_pose[0]), np.array(home_pose[1])

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
            start_target_posn, start_target_ornt,
            clearances=clearances,
            num_steps=200,
            target_link_id=target_link_id
        )
        print('\nstart_target_reached:', start_target_reached, '\n')

    if start_target_reached:
        # Lower gripper to cube level
        ready_to_grasp = move_to_target(
            start_target_posn, start_target_ornt,
            offsets=offsets, 
            num_steps=100, 
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
            start_target_posn, start_target_ornt,
            clearances=clearances,
            num_steps=100,
            target_link_id=target_link_id
            )
        print('\nitem_picked:', item_picked, '\n')

    if item_picked:
        time.sleep(1)
        # Move to target location and position end slightly above
        end_target_reached = move_to_target(
            end_target_posn, end_target_ornt, 
            clearances=clearances, 
            num_steps=300, 
            target_link_id=target_link_id
            )
        print('\nend_target_reached:', end_target_reached, '\n')

    if end_target_reached:
        time.sleep(1)
        # Lower gripper to place cube
        ready_to_release = move_to_target(
            end_target_posn, end_target_ornt, 
            # small clearance to avoid floor collision
            clearances=np.array([0., 0., 2.e-3]), 
            offsets=offsets, 
            num_steps=100, 
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
            end_target_posn, end_target_ornt, 
            clearances=clearances, 
            num_steps=100, 
            target_link_id=target_link_id
            )
        print('\nitem_placed:', item_placed, '\n')

    if item_placed:
        # Return to home position
        time.sleep(1)
        returned_home = move_to_target(
            home_posn, home_ornt, 
            num_steps=200, 
            target_link_id=target_link_id
            )
        print('\nreturned_home:', returned_home, '\n')

    if returned_home:
        # Automatically shut down after 2 min if GUI window isn't closed
        for _ in range(120 * 240):
            pbt.stepSimulation(sim1)
            time.sleep(1. / 240.)
        pbt.disconnect()
    