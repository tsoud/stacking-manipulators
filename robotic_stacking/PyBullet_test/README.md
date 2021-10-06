<h3><b>Testing and Debugging</b></h3>
Author: Tamer Abousoud<br><br>

This folder contains debugging scripts for visualizing and interacting with the robotic arm in PyBullet with GUI enabled. To run a script, open a terminal, activate a conda environment (or any virtual environment) with PyBullet and run:
```
python script_name.py
```
An interactive GUI screen will open allowing you to interact directly with the robot.<br>

- `kvG3_FK_debug.py` is a single arm that uses forward kinematics to move. Moving the sliders adjusts any of the 7 actuators or the gripper. Pressing the '1' key will print the position of the end-effector to the terminal. Pressing '0' will reset the arm and sliders to the initial positions.
- `kvG3_IK_debug.py` is a single arm that uses inverse kinematics to move. Moving the sliders adjusts the X, Y, X and roll, pitch, yaw of the target. The target is a virtual link centered between the gripper fingers (displayed as three small perpendicular lines in the GUI) for determining how to place the gripper to pick up objects. Pressing the '1' key will print the positions of the arm actuators to the terminal. Pressing '2' prints the grip force. Pressing '0' will reset the arm and sliders to the initial positions.
- `multirobot_stacking_DEBUG.py` shows 4 robot arms and a number of cubes randomly placed around them on the floor. This script helps visualize the training environment.
