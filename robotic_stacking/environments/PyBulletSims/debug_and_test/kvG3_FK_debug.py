#!/usr/bin/env python

# --------------------------------------------------------------------------- #
'''
A testing and debugging script to load a robot and test its movement using
forward kinematics.
'''
# --------------------------------------------------------------------------- #

import os
from collections import namedtuple
from importlib import resources
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
resources_path = Path('../../assets/').resolve()
robot_arm = os.path.join(resources_path, 'KinovaG3_7DOF_HandEgrip.urdf')
cube = os.path.join(resources_path, 'small_cube.urdf')

# Load a plane and cube
plane_ID = pbt.loadURDF("plane.urdf", [0, 0, 0],
	useFixedBase=True, physicsClientId=sim1)
cube_ID = pbt.loadURDF(cube, [0.3, 0.3, 0],
	useMaximalCoordinates=True, physicsClientId=sim1)
# Load the robot model.
robot_arm_ID = pbt.loadURDF(robot_arm, useFixedBase=True,
	flags=(pbt.URDF_USE_SELF_COLLISION)
	)

# Read information from the robot URDF description and get end link
n_joints = pbt.getNumJoints(robot_arm_ID)
# Find end link
joints_labeled = namedtuple('joints_labeled', ['ID', 'jointName', 'linkName'])
joint_labels = []
for jt_id in range(n_joints):
	jt_info = pbt.getJointInfo(robot_arm_ID, jt_id, sim1)
	jt_info = joints_labeled._make(list(jt_info[:2]) + [jt_info[12]])
	# print(jt_info)
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
grip_ids = [grip_L, grip_R]

# Get initial states of joints
allJoints = arm_joint_ids + grip_ids
initial_states = pbt.getJointStates(robot_arm_ID, allJoints)
initial_states = [s[0] for s in initial_states]

def add_parameters():
	joint_actions = []
	# Create parameter controls for the arm joints
	actuator1 = pbt.addUserDebugParameter('actuator1', -3.14, 3.14, 0)
	joint_actions.append(actuator1)
	actuator2 = pbt.addUserDebugParameter('actuator2', -2.41, 2.41, 0)
	joint_actions.append(actuator2)
	actuator3 = pbt.addUserDebugParameter('actuator3', -3.14, 3.14, 0)
	joint_actions.append(actuator3)
	actuator4 = pbt.addUserDebugParameter('actuator4', -2.41, 2.41, 0)
	joint_actions.append(actuator4)
	actuator5 = pbt.addUserDebugParameter('actuator5', -3.14, 3.14, 0)
	joint_actions.append(actuator5)
	actuator6 = pbt.addUserDebugParameter('actuator6', -2.41, 2.41, 0)
	joint_actions.append(actuator6)
	actuator7 = pbt.addUserDebugParameter('actuator7', -3.14, 3.14, 0)
	joint_actions.append(actuator7)
	# Parameter controls for gripper (single slider for both fingers)
	gripper = pbt.addUserDebugParameter('gripper', -3e-4, 0.03, 0)
	grip_action = [gripper] * 2

	return joint_actions, grip_action

# Enable force sensors for gripper
	pbt.enableJointForceTorqueSensor(
		robot_arm_ID, jointIndex=grip_ids[0], enableSensor=True
		)
	pbt.enableJointForceTorqueSensor(
		robot_arm_ID, jointIndex=grip_ids[1], enableSensor=True
		)

# Add the sliders to the GUI view
joint_actions, grip_action = add_parameters()

# Function to print out end-link position and orientation
def get_end_pose(end_link_id, bodyID=robot_arm_ID):
    link_state = pbt.getLinkState(bodyUniqueId=bodyID, linkIndex=target_link_id)
    position = link_state[0]
    orientation = pbt.getEulerFromQuaternion(link_state[1])
    print(f'\nPosition = {position}, Orientation = {orientation}')

# Listen for '1' keypress to print end pose.
# ('p' is already used by PyBullet GUI for something else.)
angles_key = ord('1')
# Key to reset arm to initial position
reset_key = ord('0')

# Add debugging lines to display target
loc1_id = pbt.addUserDebugLine([0., 0., 0.125], [0.025, 0., 0.125], 
	lineColorRGB=[1., 0.6, 0.2], 
	lineWidth=3.5, 
	parentObjectUniqueId=robot_arm_ID, 
	parentLinkIndex=end_effector_id
	)  # X - orange
loc2_id = pbt.addUserDebugLine([0., 0, 0.125], [0., 0.025, 0.125], 
	lineColorRGB=[1., 1., 0.2], 
	lineWidth=3.5, 
	parentObjectUniqueId=robot_arm_ID, 
	parentLinkIndex=end_effector_id
	)  # Y - yellow
loc3_id = pbt.addUserDebugLine([0., 0., 0.125], [0., 0., 0.15], 
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

# Set GUI camera for better view
pbt.resetDebugVisualizerCamera(
	cameraDistance=1.4, 
	cameraYaw=45, 
	cameraPitch=-42, 
	cameraTargetPosition=[0.25, 0, 0]
	)

while True:
	pbt.configureDebugVisualizer(pbt.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
	pbt.configureDebugVisualizer(pbt.COV_ENABLE_SHADOWS, 0)
	# Read desired actuator angles from sliders
	jt_positions = [pbt.readUserDebugParameter(pos) for pos in joint_actions]
	# Set actuator positions to move arm
	pbt.setJointMotorControlArray(
		bodyUniqueId=robot_arm_ID, 
		jointIndices=arm_joint_ids, 
		controlMode=pbt.POSITION_CONTROL, 
		targetPositions=jt_positions)
	# Set gripper directly
	gripper_position = [pbt.readUserDebugParameter(g) for g in grip_action]
	pbt.setJointMotorControlArray(
		bodyUniqueId=robot_arm_ID, 
		jointIndices=grip_ids, 
		controlMode=pbt.POSITION_CONTROL, 
		targetPositions=gripper_position, 
		forces=[185, 185]  # max gripper force
		)
	# Print actuator positions if '1' key is pressed
	keys = pbt.getKeyboardEvents()
	if angles_key in keys and (keys.get(angles_key) & pbt.KEY_WAS_TRIGGERED):
		get_end_pose(end_effector_id)
	# Reset actuators to initial position if '0' is pressed
	if reset_key in keys and (keys.get(reset_key) & pbt.KEY_WAS_TRIGGERED):
		pbt.removeAllUserParameters(sim1)
		[pbt.resetJointState(robot_arm_ID, j, s) \
			for j, s in zip(allJoints, initial_states)]
		# Reset sliders
		joint_actions, grip_action = add_parameters()

	pbt.stepSimulation(sim1)

pbt.disconnect()
