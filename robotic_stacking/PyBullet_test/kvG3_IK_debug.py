#!/usr/bin/env python

# --------------------------------------------------------------------------- #
'''
A testing and debugging script to load a robot, a test object (1kg cube), and
test the robot's movement and gripping action using inverse kinematics.
'''
# --------------------------------------------------------------------------- #

import os
from collections import namedtuple
from pathlib import Path

import numpy as np
import pybullet as pbt
import pybullet_data

# --------------------------------------------------------------------------- #

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

# Load a plane and cube
floor_ID = pbt.loadURDF("plane.urdf", [0, 0, 0],
                        useFixedBase=True, physicsClientId=sim1)
# Load a small cube
cube_ID = pbt.loadURDF(cube, [0.3, 0.3, 0.03],
                       useMaximalCoordinates=True, physicsClientId=sim1)
# Load the robot model.
robot_arm_ID = pbt.loadURDF(robot_arm, useFixedBase=True,
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
arm_joint_ids = [j.ID for j in joint_labels[:end_effector_id + 1]]
n_active_dofs = len(arm_joint_ids)
grip_ids = [grip_L, grip_R]

# Get initial state of end-effector
end_eff_initial_state = pbt.getLinkState(robot_arm_ID, target_link_id)[:2]
end_initial_pos = end_eff_initial_state[0]
X_0, Y_0, Z_0 = end_initial_pos[0], end_initial_pos[1], end_initial_pos[2]
end_initial_ornt = pbt.getEulerFromQuaternion(end_eff_initial_state[1])
P_0, Yw_0, R_0 = end_initial_ornt[0], end_initial_ornt[1], end_initial_ornt[2]
print('End effector initial position:', end_initial_pos,
      '\nEnd effector initial orientation:', end_initial_ornt, end='\n\n')


def add_IK_parameters():
    # Arrays to collect desired position and orientation end-effector
    end_effector_position = []
    end_effector_orientation = []
    # Create parameter controls for the end effector
    position_X = pbt.addUserDebugParameter('end_X', -1.5, 1.5, X_0)
    end_effector_position.append(position_X)
    position_Y = pbt.addUserDebugParameter('end_Y', -1.5, 1.5, Y_0)
    end_effector_position.append(position_Y)
    position_Z = pbt.addUserDebugParameter('end_Z', -1.5, 1.5, Z_0)
    end_effector_position.append(position_Z)
    orientation_P = pbt.addUserDebugParameter('pitch', -3.14, 3.14, P_0)
    end_effector_orientation.append(orientation_P)
    orientation_Y = pbt.addUserDebugParameter('yaw', -3.14, 3.14, Yw_0)
    end_effector_orientation.append(orientation_Y)
    orientation_R = pbt.addUserDebugParameter('roll', -3.14, 3.14, R_0)
    end_effector_orientation.append(orientation_R)
    # Controls for gripping
    gripper = pbt.addUserDebugParameter('gripper', -3e-4, 0.03, 0)
    # Add gripper twice (once for each finger)
    grip_action = [gripper] * 2

    return end_effector_position, end_effector_orientation, grip_action


# Enable force sensors for gripper
pbt.enableJointForceTorqueSensor(
    robot_arm_ID, jointIndex=grip_ids[0], enableSensor=True
)
pbt.enableJointForceTorqueSensor(
    robot_arm_ID, jointIndex=grip_ids[1], enableSensor=True
)

# Add the sliders to the GUI view
end_effector_position, end_effector_orientation, grip_action \
    = add_IK_parameters()

# Print out some robot info
print(joint_labels)
print(f'\nEnd effector ID: {end_effector_id}\n')

# Function to print out arm actuator positions
def get_actuator_angles(joint_ids, bodyID=robot_arm_ID):
    states = pbt.getJointStates(bodyID, joint_ids)
    [print(f'Actuator{i + 1} = {states[i][0]}') for i in range(len(states))]
    print('\n')

# Print grip force (reaction force on joint when gripping)
def get_grip_force(grip_ids=grip_ids, bodyID=robot_arm_ID):
    states = pbt.getJointStates(bodyID, grip_ids)
    # read actuator force
    print(f'grip_L = {states[0][3]}, grip_R = {states[1][3]}')
    print('\n')

# Get initial states of joints
allJoints = arm_joint_ids + grip_ids
initial_states = pbt.getJointStates(robot_arm_ID, allJoints)
initial_states = [s[0] for s in initial_states]

# Listen for '1' keypress to print actuator angles.
# ('p' is already used by PyBullet GUI for something else.)
angles_key = ord('1')
# Key to print grip force
gripF_key = ord('2')
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

# Remove shadows from display for better performance
pbt.configureDebugVisualizer(pbt.COV_ENABLE_SHADOWS, 0)

while True:
    pbt.configureDebugVisualizer(pbt.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
    # Read parameters from sliders
    target_position = [pbt.readUserDebugParameter(
        p) for p in end_effector_position]
    target_orientatn = [pbt.readUserDebugParameter(
        o) for o in end_effector_orientation]
    target_orientatn = pbt.getQuaternionFromEuler(target_orientatn)
    # Use IK solver to calculate joint angles
    joint_angles = pbt.calculateInverseKinematics(
        bodyUniqueId=robot_arm_ID,
        endEffectorLinkIndex=target_link_id,
        targetPosition=target_position,
        targetOrientation=target_orientatn
    )
    # Set the joint angles to move the end-effector
    pbt.setJointMotorControlArray(
        bodyUniqueId=robot_arm_ID,
        jointIndices=arm_joint_ids,
        controlMode=pbt.POSITION_CONTROL,
        targetPositions=joint_angles[:n_active_dofs]
    )
    # Set gripper directly
    gripper_position = [pbt.readUserDebugParameter(g) for g in grip_action]
    pbt.setJointMotorControlArray(
        bodyUniqueId=robot_arm_ID,
        jointIndices=grip_ids,
        controlMode=pbt.POSITION_CONTROL,
        targetPositions=gripper_position,
        forces=[185, 185]
    )

    # Print actuator positions if '1' key is pressed
    keys = pbt.getKeyboardEvents()
    if angles_key in keys and (keys.get(angles_key) & pbt.KEY_WAS_TRIGGERED):
            get_actuator_angles(arm_joint_ids)
        # Print gripper force if '2' key is pressed
    if gripF_key in keys and (keys.get(gripF_key) & pbt.KEY_WAS_TRIGGERED):
            get_grip_force()
        # Reset actuators to initial position if '0' is pressed
    if reset_key in keys and (keys.get(reset_key) & pbt.KEY_WAS_TRIGGERED):
            pbt.removeAllUserParameters(sim1)
            [pbt.resetJointState(robot_arm_ID, j, s)
             for j, s in zip(allJoints, initial_states)]
            # Reset sliders
            end_effector_position, end_effector_orientation, grip_action \
                = add_IK_parameters()

    pbt.stepSimulation(sim1)
