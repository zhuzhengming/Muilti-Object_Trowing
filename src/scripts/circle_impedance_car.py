import sys

import matplotlib

sys.path.append('../')
import copy
import math
import numpy as np
import time
from utils.controller_utils import Robot
import rospy
import tools.rotations as rot
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')


def generate_circle_via_points(current_pose, radius=0.05, num_points=10):
    x0, y0, z0 = current_pose[:3]
    center_x = x0 - radius
    center_y = y0
    center_z = z0

    theta_array = np.linspace(0, 2*math.pi, num_points)

    via_points = np.zeros((num_points, 7))
    for i, theta in enumerate(theta_array):
        # x = center_x + radius*cos(theta), y = center_y + radius*sin(theta)
        x = center_x + radius * math.cos(theta)
        y = center_y + radius * math.sin(theta)
        z = center_z
        via_points[i, :3] = np.array([x, y, z])

        via_points[i, 3:] = current_pose[3:]
    return via_points


def calculate_error(current_pose, target_pose):
    position_error = np.linalg.norm(current_pose[:3] - target_pose[:3])
    return position_error


def plot_trajectory_and_error(via_points, errors, robot_positions):
    target_x = via_points[:, 0]
    target_y = via_points[:, 1]
    target_z = via_points[:, 2]

    robot_x = [pos[0] for pos in robot_positions]
    robot_y = [pos[1] for pos in robot_positions]
    robot_z = [pos[2] for pos in robot_positions]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

    ax1.plot(errors, 'r-', label='Position Error')
    ax1.set_ylabel('Position Error (meters)', color='r')
    ax1.tick_params(axis='y', labelcolor='r')
    ax1.set_title('Position Error and Z Trajectory')

    ax2 = ax1.twinx()
    ax2.plot(target_z, 'b--', label='Target z Trajectory')
    ax2.plot(robot_z, 'b-', label='Robot z Trajectory')
    ax2.set_ylabel('Z Position (meters)', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2.legend(loc='upper right')

    ax1.set_ylim(min(errors) - 0.1, max(errors) + 0.1)
    ax2.set_ylim(min(min(target_z), min(robot_z)) - 0.1, max(max(target_z), max(robot_z)) + 0.1)

    ax2 = fig.add_subplot(212)
    ax2.plot(target_x, target_y, 'b--', label='Target Trajectory')
    ax2.plot(robot_x, robot_y, 'g-', label='Robot Trajectory')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Position (meters)')
    ax2.set_aspect('equal', 'box')
    ax2.legend(loc='upper right')

    ax2.set_ylim(min(min(target_y), min(robot_y)) - 0.1, max(max(target_y), max(robot_y)) + 0.1)

    plt.grid()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    r = Robot(optitrack_frame_names=['iiwa_base7', 'realsense_m'], position_control=False)


    current_pose_0 = r.x
    via_points = generate_circle_via_points(current_pose_0, radius=0.15, num_points=20)

    errors = []
    robot_positions = []

    for target_pose in via_points:
        r.iiwa_cartesion_impedance_control(target_pose)
        current_pose = r.x
        error = calculate_error(current_pose, target_pose)
        errors.append(error)
        robot_positions.append(current_pose[:3])


    plot_trajectory_and_error(via_points, errors, robot_positions)