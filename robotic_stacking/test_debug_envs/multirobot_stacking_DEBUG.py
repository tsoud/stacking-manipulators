#!/usr/bin/env python

# --------------------------------------------------------------------------- #
'''
A multi-robot testing and debugging script with 4 robotic manipulators and 
22 cubes.
'''
# --------------------------------------------------------------------------- #

import os
from collections import namedtuple
from pathlib import Path

import numpy as np
import pybullet as pbt
import pybullet_data
from shapely.geometry import Point, MultiPoint, Polygon, MultiPolygon

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
resources_path = Path('../assets').resolve()
robot_arm = os.path.join(resources_path, 'KinovaG3_7DOF_HandEgrip.urdf')
cube = os.path.join(resources_path, 'small_cube.urdf')

# Load the "floor" plane
floor_ID = pbt.loadURDF("plane.urdf", [0, 0, 0], 
	useFixedBase=True, physicsClientId=sim1)

# Load the four robots
NUM_ROBOTS = 4
base_positions = np.array([[1., 0., 0.], [-1., 0., 0.], 
						   [0., 1., 0.], [0., -1., 0.]])
# Orientations in axis-angle
base_orientations = np.array([[0., 0., 1., np.pi], 
							  [0., 0., 1., 0], 
						      [0., 0., 1., -(np.pi / 2)], 
							  [0., 0., 1., (np.pi / 2)]])

robots = {}
for n in range(NUM_ROBOTS):
	orientation = pbt.getQuaternionFromAxisAngle(base_orientations[n, :-1], 
		base_orientations[n, -1])
	robots[f'arm{n + 1}_id'] = pbt.loadURDF(robot_arm, 
		basePosition=base_positions[n, :], 
		baseOrientation=orientation,
		useFixedBase=True, 
		flags=(pbt.URDF_USE_SELF_COLLISION), 
		physicsClientId=sim1
		)
	# Label the arm on the screen
	txt_offset = base_positions[n, :] + np.array([0.05] * 3)
	label_viz_id = pbt.addUserDebugText(str(n + 1), 
		txt_offset, 
		textColorRGB=[1, 0, 1], 
		textSize=2.0
		)

# Load cubes into environment
NUM_CUBES = 22
RNG_SEED = 12345
# Seed the random generator
np.random.seed(RNG_SEED)
# Areas to avoid placing cubes (close to robot bases)
invalid_locs = MultiPolygon(
	[Polygon([(0.9, -0.1), (1.1, -0.1), (1.1, 0.1), (0.9, 0.1)]), 
	Polygon([(-0.9, -0.1), (-1.1, -0.1), (-1.1, 0.1), (-0.9, 0.1)]), 
	Polygon([(-0.1, 0.9), (-0.1, 1.1), (0.1, 1.1), (0.1, 0.9)]), 
	Polygon([(-0.1, -0.9), (-0.1, -1.1), (0.1, -1.1), (0.1, -0.9)])
	]
)
# Create a grid (40 x 40 here) of candidate points for placing cubes
X_c = np.linspace(-1.22, 1.22, num=40)
Y_c = np.linspace(-1.22, 1.22, num=40)
XY_c = np.transpose([np.tile(X_c, len(Y_c)), np.repeat(Y_c, len(X_c))])
# Avoid invalid locations
points = MultiPoint(XY_c)
valid_points = MultiPoint([p for p in points if not invalid_locs.contains(p)])
valid_points = np.asarray(valid_points)
# Select cube positions
cube_posn_ids = np.random.choice(np.arange(len(valid_points)), 
	size=NUM_CUBES, replace=False)

cubes = {}
for n in cube_posn_ids:
	position = np.append(valid_points[n], np.array([0]))
	angle = np.random.uniform(0., np.pi/2)
	orientation = pbt.getQuaternionFromAxisAngle([0., 0., 1.], angle)
	cubes[f'cube{n + 1}_id'] = pbt.loadURDF(cube, 
		basePosition=position, 
		baseOrientation=orientation, 
		useMaximalCoordinates=True, 
		physicsClientId=sim1
		)

# Set GUI camera for better view
pbt.resetDebugVisualizerCamera(
	cameraDistance=1.6, 
	cameraYaw=150, 
	cameraPitch=-47, 
	cameraTargetPosition=[0, 0, 0]
	)

view_matrix = pbt.computeViewMatrixFromYawPitchRoll(
	cameraTargetPosition=[0, 0, 0], 
	distance=1.5, 
	yaw=89.9, pitch=-90.1, roll=0., 
	upAxisIndex=2
)
proj_matrix = pbt.computeProjectionMatrixFOV(
	fov=60, aspect=(960 / 720), nearVal=0.1, farVal=100.
)
pbt.getCameraImage(
	960, 720, 
	viewMatrix=view_matrix, 
	projectionMatrix=proj_matrix, 
	renderer=pbt.ER_BULLET_HARDWARE_OPENGL
)

while True:
	pbt.configureDebugVisualizer(pbt.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
	pbt.configureDebugVisualizer(pbt.COV_ENABLE_SHADOWS, 0)
	pbt.stepSimulation(sim1)

pbt.disconnect(sim1)