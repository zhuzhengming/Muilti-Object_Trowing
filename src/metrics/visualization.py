#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import atexit
import matplotlib
matplotlib.use('Qt5Agg')


class TrajectoryVisualizer:
    def __init__(self):
        rospy.init_node('pose_3d_recorder')

        self.positions = []

        self.sub = rospy.Subscriber(
            '/vrpn_client_node/cube_z/pose_from_iiwa_7_base',
            PoseStamped,
            self.callback,
            queue_size=100
        )

        atexit.register(self.on_exit)

    def callback(self, msg):
        self.positions.append([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])

    def save_data(self, filename='../output/data/optitrack.npy'):
        np.save(filename, np.array(self.positions))
        print(f"Saved {len(self.positions)} points to {filename}")

    def plot_3d_trajectory(self):
        if len(self.positions) < 2:
            print("Not enough points to plot (need at least 2 points)")
            return

        pos = np.array(self.positions)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'b-', linewidth=1, label='Trajectory')

        ax.scatter(pos[0, 0], pos[0, 1], pos[0, 2], c='g', s=100, label='Start')
        ax.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2], c='r', s=100, label='End')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D Trajectory')
        ax.legend()

        max_range = np.array([
            pos[:, 0].max() - pos[:, 0].min(),
            pos[:, 1].max() - pos[:, 1].min(),
            pos[:, 2].max() - pos[:, 2].min()
        ]).max() / 2.0

        mid_x = (pos[:, 0].max() + pos[:, 0].min()) * 0.5
        mid_y = (pos[:, 1].max() + pos[:, 1].min()) * 0.5
        mid_z = (pos[:, 2].max() + pos[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        plt.tight_layout()
        plt.show()

    def on_exit(self):
        self.save_data()
        self.plot_3d_trajectory()


if __name__ == '__main__':
    recorder = TrajectoryVisualizer()
    rospy.spin()