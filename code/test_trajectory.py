import os
from pr_classes import RobotTrajectory
from constants import DATASET_PATH, TRAJ_FILE


traj_input_file = os.path.join(DATASET_PATH, TRAJ_FILE)
trajectory = RobotTrajectory(traj_input_file)
trajectory.plotting()