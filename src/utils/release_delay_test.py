import sys

sys.path.append('../')
import numpy as np
from utils.controller_utils import Robot
import rospy
from geometry_msgs.msg import WrenchStamped
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Qt5Agg')
import time
import atexit

PLOT_ON_EXIT = True


def hand_control(recorder):
    r = Robot(optitrack_frame_names=['iiwa_base7', 'realsense_m'], position_control=True)

    home_pose = np.array([ 0.01186698, -0.31076046, -0.24081303, -0.29323739, -0.42955531,
        0.83253228,  0.68963376,  0.28323768, -0.06243891, -0.26657226,
       -0.02011721, -0.26511309,  1.0921544 ,  0.56434586, -0.30931694,
        0.85974391])

    envelop_pose = np.array([0.02261951, -0.30808927, -0.19149993, -0.29282156, -0.34230519,
                             1.12686636, 0.90202891, 0.40875289, -0.06173849, -0.26592787,
                             0.00727593, -0.26036765, 1.11862433, 0.54512895, 0.20616064,
                             0.98999667])

    while True:
        # Move to envelop pose
        r.move_to_joints(envelop_pose, vel=[0.2, 8.0])
        time.sleep(3)

        # Move to home pose and record event
        recorder.record_event('home')
        r.move_to_joints(home_pose, vel=[0.2, 8.0])
        time.sleep(3)


class FTDataRecorder:
    def __init__(self):
        self.force_z_data = []
        self.event_data = []
        self.start_time = time.time()
        self.sub = rospy.Subscriber(
            '/ft_sensor/netft_data',
            WrenchStamped,
            self.wrench_callback
        )
        rospy.loginfo("Initialized FT Data Recorder")
        atexit.register(self.on_exit)

    def wrench_callback(self, msg):
        z_force = msg.wrench.force.z
        timestamp = time.time() - self.start_time
        self.force_z_data.append((timestamp, z_force))

    def record_event(self, event_type):
        timestamp = time.time() - self.start_time
        self.event_data.append((timestamp, event_type))
        rospy.loginfo(f"Recorded {event_type} event at {timestamp:.3f}s")

    def save_data(self):
        force_filename = '../output/data/release_data/release_delay.npy'
        event_filename = '../output/data/release_data/event_data.npy'

        if self.force_z_data:
            np.save(force_filename, np.array(self.force_z_data))
        if self.event_data:
            np.save(event_filename, np.array(self.event_data))
        rospy.loginfo(f"Saved {len(self.force_z_data)} force samples and {len(self.event_data)} events")

    def plot_data(self):
        if not self.force_z_data and not self.event_data:
            rospy.logwarn("No data to plot")
            return

        plt.figure(figsize=(12, 6))

        if self.force_z_data:
            times, values = zip(*self.force_z_data)
            plt.plot(times, values, 'b-', lw=1, label='Z-axis Force')

        if self.event_data:
            home_times = [t for t, e in self.event_data if e == 'home']
            if home_times:
                y_min = min(values) if self.force_z_data else 0
                y_max = max(values) if self.force_z_data else 0
                plt.vlines(home_times, y_min, y_max,
                           colors='r', linestyles='--', lw=1.5,
                           label='Release Command')

        plt.title('Force Measurement with Control Events')
        plt.xlabel('Time since start (s)')
        plt.ylabel('Force (N)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def on_exit(self):
        self.save_data()
        if PLOT_ON_EXIT:
            self.plot_data()


if __name__ == '__main__':
    recorder = FTDataRecorder()
    try:
        hand_control(recorder)
    except KeyboardInterrupt:
        pass
    finally:
        recorder.save_data()