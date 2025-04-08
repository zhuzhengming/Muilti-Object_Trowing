import sys
sys.path.append('../')
import numpy as np
from utils.controller_utils import Robot
import rospy
import rospy
from geometry_msgs.msg import WrenchStamped
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
import time
import atexit

PLOT_ON_EXIT = True

def hand_control():
    r = Robot(optitrack_frame_names=['iiwa_base7', 'realsense_m'], position_control=True)

    home_pose = np.array([ 0.00487308, -0.3105059 , -0.24132268, -0.29216679, -0.48376039,
        0.73783028,  0.80553937,  0.36848938, -0.06286451, -0.26625116,
       -0.02076643, -0.26384511,  1.11435883,  0.39652053, -0.19789294,
        0.90250333])

    envelop_pose = np.array([ 0.02261951, -0.30808927, -0.19149993, -0.29282156, -0.34230519,
        1.12686636,  0.90202891,  0.40875289, -0.06173849, -0.26592787,
        0.00727593, -0.26036765,  1.11862433,  0.54512895,  0.20616064,
        0.98999667])

    while (True):
        r.move_to_joints(envelop_pose, vel=[0.2,8.0])
        time.sleep(1)
        r.move_to_joints(home_pose, vel=[0.2,8.0])
        time.sleep(1)


class FTDataRecorder:
    def __init__(self):
        self.force_z_data = []
        self.statr_time = time.time()
        self.sub = rospy.Subscriber(
            '/ft_sensor/netft_data',
            WrenchStamped,
            self.wrench_callback
        )
        rospy.loginfo("Initialized FT Data Recorder")
        atexit.register(self.on_exit)

    def wrench_callback(self, msg):
        z_force = msg.wrench.force.z

        timestamp = time.time() - self.statr_time
        self.force_z_data.append((timestamp, z_force))

    def save_data(self, filename='../output/data/release_delay.npy'):
        np.save(filename, np.array(self.force_z_data))
        rospy.loginfo(f"Saved {len(self.force_z_data)} samples to {filename}")

    def plot_data(self):
        if not self.force_z_data:
            rospy.logwarn("No data to plot")
            return

        times, values = zip(*self.force_z_data)
        plt.figure(figsize=(10, 5))
        plt.plot(times, values, 'b-', linewidth=1)
        plt.title('Force Z Data (Relative Time)')
        plt.xlabel('Time since start (seconds)')
        plt.ylabel('Force (N)')
        plt.grid(True)
        plt.show()

    def on_exit(self):
        self.save_data()
        if PLOT_ON_EXIT:
            self.plot_data()


if __name__ == '__main__':
    recorder = FTDataRecorder()
    hand_control()

