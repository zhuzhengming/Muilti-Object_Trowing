import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
sys.path.append("../scripts")
from hedgehog import VelocityHedgehog
matplotlib.use('Qt5Agg')

class TrackingEvaluation:
    def __init__(self, filepath, robot_path):
        self.data = np.load(filepath, allow_pickle=True).item()
        q_min = np.array([-2.96705972839, -2.09439510239, -2.96705972839, -2.09439510239, -2.96705972839,
                          -2.09439510239, -3.05432619099])
        q_max = np.array([2.96705972839, 2.09439510239, 2.96705972839, 2.09439510239, 2.96705972839,
                          2.09439510239, 3.05432619099])
        q_dot_max = np.array([1.71, 1.74, 1.745, 2.269, 2.443, 3.142, 3.142])
        q_dot_min = -q_dot_max

        self.robot = VelocityHedgehog(q_min, q_max, q_dot_min, q_dot_max, robot_path, model_exist=True)

        self.timestamp = np.array(self.data['stamp'])
        self.actual_position = np.array(self.data['real_pos'])
        self.actual_velocity = np.array(self.data['real_vel'])
        self.target_position = np.array(self.data['target_pos'])
        self.target_velocity = np.array(self.data['target_vel'])

    def plot_joint_tracking(self):
        """Plot joint position and velocity tracking for all 7 joints"""
        fig, axs = plt.subplots(7, 2, figsize=(14, 20))

        for joint_idx in range(7):
            # Plot position tracking
            axs[joint_idx, 0].plot(self.timestamp, self.actual_position[:, joint_idx],
                                   label='Actual', color='blue')
            axs[joint_idx, 0].plot(self.timestamp, self.target_position[:, joint_idx],
                                   label='Target', linestyle='--', color='green')
            axs[joint_idx, 0].set_ylabel(f'Joint {joint_idx + 1}\nPosition (rad)', fontsize=8)
            axs[joint_idx, 0].grid(True)
            axs[joint_idx, 0].tick_params(axis='both', labelsize=8)

            # Plot velocity tracking
            axs[joint_idx, 1].plot(self.timestamp, self.actual_velocity[:, joint_idx],
                                   label='Actual', color='blue')
            axs[joint_idx, 1].plot(self.timestamp, self.target_velocity[:, joint_idx],
                                   label='Target', linestyle='--', color='green')
            axs[joint_idx, 1].set_ylabel(f'Joint {joint_idx + 1}\nVelocity (rad/s)', fontsize=8)
            axs[joint_idx, 1].grid(True)
            axs[joint_idx, 1].tick_params(axis='both', labelsize=8)

            # Add legend to first row only
            if joint_idx == 0:
                axs[joint_idx, 0].legend(loc='upper right', fontsize=6)
                axs[joint_idx, 1].legend(loc='upper right', fontsize=6)

        # Set common xlabel
        for ax in axs[-1, :]:
            ax.set_xlabel('Time (s)', fontsize=8)

        plt.tight_layout()
        plt.suptitle('Joint Space Tracking Performance', y=1.02)
        plt.show()

    def plot_ee_tracking(self):
        """Plot end-effector trajectories in 3D space"""
        # Calculate trajectories
        actual_pos_ee, actual_vel_ee = [], []
        target_pos_ee, target_vel_ee = [], []

        for i in range(len(self.timestamp)):
            # Actual trajectory
            a_pos, J_act = self.robot.forward(self.actual_position[i])
            actual_pos_ee.append(a_pos)
            actual_vel_ee.append(J_act @ self.actual_velocity[i])

            # Target trajectory
            t_pos, J_tar = self.robot.forward(self.target_position[i])
            target_pos_ee.append(t_pos)
            target_vel_ee.append(J_tar @ self.target_velocity[i])

        # Convert to numpy arrays
        actual_pos_ee = np.array(actual_pos_ee)
        actual_vel_ee = np.array(actual_vel_ee)
        target_pos_ee = np.array(target_pos_ee)
        target_vel_ee = np.array(target_vel_ee)

        # Create figure with 3x2 subplots for components
        fig, axs = plt.subplots(3, 2, figsize=(14, 12))
        axes_labels = ['X', 'Y', 'Z']
        colors = ['#1f77b4', '#ff7f0e']

        # Plot position and velocity components
        for i in range(3):
            # Position tracking
            axs[i, 0].plot(self.timestamp, actual_pos_ee[:, i],
                           color=colors[0], linewidth=1.5, label='Actual')
            axs[i, 0].plot(self.timestamp, target_pos_ee[:, i],
                           color=colors[1], linestyle='--', linewidth=1.2, label='Target')
            axs[i, 0].set_ylabel(f'{axes_labels[i]} Position (m)', fontsize=10)
            axs[i, 0].grid(True, alpha=0.3)

            # Velocity tracking
            axs[i, 1].plot(self.timestamp, actual_vel_ee[:, i],
                           color=colors[0], linewidth=1.5, label='Actual')
            axs[i, 1].plot(self.timestamp, target_vel_ee[:, i],
                           color=colors[1], linestyle='--', linewidth=1.2, label='Target')
            axs[i, 1].set_ylabel(f'{axes_labels[i]} Velocity (m/s)', fontsize=10)
            axs[i, 1].grid(True, alpha=0.3)

            # Add legend to first row only
            if i == 0:
                axs[i, 0].legend(loc='upper right', fontsize=9, framealpha=0.9)
                axs[i, 1].legend(loc='upper right', fontsize=9, framealpha=0.9)

        # Set common xlabel
        for row in axs:
            for ax in row:
                ax.tick_params(axis='both', labelsize=9)
        axs[-1, 0].set_xlabel('Time (s)', fontsize=10)
        axs[-1, 1].set_xlabel('Time (s)', fontsize=10)

        plt.tight_layout()
        plt.suptitle('End-Effector Component-wise Tracking Performance', y=1.02, fontsize=12)
        plt.show()


        # Create 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot trajectories
        line_actual = ax.plot3D(actual_pos_ee[:, 0], actual_pos_ee[:, 1], actual_pos_ee[:, 2],
                                label='Actual Trajectory', color='blue', linewidth=2)
        line_target = ax.plot3D(target_pos_ee[:, 0], target_pos_ee[:, 1], target_pos_ee[:, 2],
                                label='Target Trajectory', color='red', linestyle='--', linewidth=2)

        # Plot start and end points
        ax.scatter(actual_pos_ee[0, 0], actual_pos_ee[0, 1], actual_pos_ee[0, 2], c='red', label='Start')
        ax.scatter(actual_pos_ee[-1, 0], actual_pos_ee[-1, 1], actual_pos_ee[-1, 2], c='blue', label='End (Actual)')
        ax.scatter(target_pos_ee[-1, 0], target_pos_ee[-1, 1], target_pos_ee[-1, 2], c='green', label='End (Target)')

        # Configure axes
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
        ax.set_zlabel('Z (m)', fontsize=10)
        ax.set_title('End-Effector 3D Trajectory Comparison', pad=20)

        # Add legend and grid
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True)

        # Equal axis aspect ratio
        max_range = np.array([actual_pos_ee.max() - actual_pos_ee.min(),
                              target_pos_ee.max() - target_pos_ee.min()]).max() / 5
        mid_x = (actual_pos_ee[:, 0].max() + actual_pos_ee[:, 0].min()) * 0.5
        mid_y = (actual_pos_ee[:, 1].max() + actual_pos_ee[:, 1].min()) * 0.5
        mid_z = (actual_pos_ee[:, 2].max() + actual_pos_ee[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Set view angle
        ax.view_init(elev=25, azim=45)

        plt.tight_layout()
        plt.show()

def plot_tracking_data(file_path):
    data = np.load(file_path, allow_pickle=True).item()

    kp_candidates = data['kp_candidates']  # 1D array
    kd_candidates = data['kd_candidates']  # 1D array
    pos_error_sum = data['pos_error']  # 2D array
    vel_error_sum = data['vel_error']  # 2D array

    kp_mesh, kd_mesh = np.meshgrid(kp_candidates, kd_candidates)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')

    surface1 = ax1.plot_surface(kp_mesh, kd_mesh, pos_error_sum, edgecolor='none')
    ax1.set_xlabel('Kp')
    ax1.set_ylabel('Kd')
    ax1.set_zlabel('Position Tracking Error')

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    surface2 = ax2.plot_surface(kp_mesh, kd_mesh, vel_error_sum, edgecolor='none')
    ax2.set_xlabel('Kp')
    ax2.set_ylabel('Kd')
    ax2.set_zlabel('Velocity Tracking Error')

    plt.show()

if __name__ == '__main__':
    filepath = f'../output/data/throwing.npy'
    robot_path = '../description/iiwa7_allegro_throwing.xml'
    evaluator = TrackingEvaluation(filepath, robot_path)

    # Plot joint space tracking
    evaluator.plot_joint_tracking()

    # Plot Cartesian space tracking
    evaluator.plot_ee_tracking()
