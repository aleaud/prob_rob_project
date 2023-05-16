# Planar Monocular SLAM

This is the code for the project part of the Probabilistic Robotics exam. The code was written in Python because I've felt more comfortable and the `Numpy` library provides a lot of helping built-in functions and tools (such as SVD, solve linear system...)

The goal of this project is to develop a SLAM system using Total Least Square algorithm to determine the trajectory of a robot moving in a given environment. This robot perceives the environment only using a stereo camera mounted on its base and its motion by the odometry sensor.

The dataset provides ground truth poses and landmarks to be used for evaluation, along with poses coming from the noisy odometry and measurements (i.e. image points) with data association. First, the program starts by loading the data about the world (set of 3D landmarks) and the trajectory followed by the robot, then the landmark positions were estimated either via triangulation or using correspondences from all images. Then the associations for poses and landmark projections are built and all that was the set of arguments to perform the Total Least Squares method.

## How to run

Run with `make tls-pc` to estimate landmark and robot poses using all image correspondences coming from the observations. It is suggested to run `make clean` before testing.
