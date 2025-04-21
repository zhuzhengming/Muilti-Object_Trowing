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

    home_pose = np.array([ -0.11325745,  0.13721696,  0.73047104,  0.19944343, -0.36932006,
        0.83361463,  0.91679593,  0.52381956, -0.21987901,  0.38563108,
        0.51842488,  0.        ,  1.20761276,  0.28859433,  0.38003315,
        1.15945389])

    envelop_pose = np.array([0.3794, 0.1122, 0.8395, 0.2504, -0.5161, 0.7940, 0.9335,
                0.5431, -0.4874, 0.3151, 0.5176, 0.4811, 1.0792, 0.2427,
                0.2730, 1.1701])

    while True:
        # Move to envelop pose
        r.move_to_joints(envelop_pose, vel=[0.2, 8.0])
        time.sleep(3)

        # Move to home pose and record event
        recorder.record_event('home')
        r.move_to_joints(home_pose, vel=[0.2, 8.0])
        time.sleep(3)


class FTDataRecorder:
    def __init__(self, release_delay_path=None, event_path=None):
        self.is_loading_mode = False

        if release_delay_path and event_path:
            try:
                self.force_z_data = np.load(release_delay_path, allow_pickle=True)
                self.event_data = np.load(event_path, allow_pickle=True)

                if self.force_z_data.dtype == object:
                    self.force_z_data = np.array([list(x) for x in self.force_z_data])
                if self.event_data.dtype == object:
                    self.event_data = np.array([list(x) for x in self.event_data])

                self.is_loading_mode = True
                rospy.loginfo("Loaded historical data")

            except FileNotFoundError:
                rospy.logwarn("Data files not found, starting fresh")
                self.force_z_data = []
                self.event_data = []
        else:
            self.force_z_data = []
            self.event_data = []

        if not self.is_loading_mode:
            self.start_time = time.time()
            self.sub = rospy.Subscriber(
                '/ft_sensor/netft_data',
                WrenchStamped,
                self.wrench_callback
            )
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
        force_filename = '../output/data/release_data/posture_2_release_delay.npy'
        event_filename = '../output/data/release_data/posture_2_event_data.npy'

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

    def plot_exist_data(self):
        if self.force_z_data.ndim != 2 or self.force_z_data.shape[1] != 2:
            rospy.logerr("Invalid force data format")
            return

        times = self.force_z_data[:, 0]
        values = -self.force_z_data[:, 1]

        if self.event_data.ndim == 2 and self.event_data.shape[1] == 2:
            home_times = self.event_data[self.event_data[:, 1] == 'home'][:, 0]
        else:
            rospy.logwarn("Event data format invalid")
            home_times = []

        plt.figure(figsize=(10, 5))

        plt.plot(times, values, 'b-', lw=1.5, label='Force (N)')

        if len(home_times) > 0:
            y_min, y_max = np.min(values) * 0.9, np.max(values) * 1.1
            plt.vlines(home_times, y_min, y_max,
                       colors='r', linestyles='--', lw=1.5,
                       label='Release Events')

        plt.xlim(left=0)

        plt.title('Force Measurement with Release Events')
        plt.xlabel('Original Timestamp (s)')
        plt.ylabel('Force (N)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def on_exit(self):
        self.save_data()
        if PLOT_ON_EXIT:
            self.plot_data()


if __name__ == '__main__':
    release_delay_path = '../output/data/release_data/posture_2_release_delay.npy'
    event_data_path = '../output/data/release_data/posture_2_event_data.npy'
    recorder = FTDataRecorder()
    # recorder.plot_exist_data()

    try:
        hand_control(recorder)
    except KeyboardInterrupt:
        pass
    finally:
        recorder.save_data()
