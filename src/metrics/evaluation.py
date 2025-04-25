import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
sys.path.append("../scripts")
from hedgehog import VelocityHedgehog
matplotlib.use('Qt5Agg')
import os
import pandas as pd
from scipy.spatial.distance import cdist

class TrackingEvaluation:
    def __init__(self, robot_path, filepath=None, batch_path=None, posture='posture1'):
        q_min = np.array([-2.96705972839, -2.09439510239, -2.96705972839, -2.09439510239, -2.96705972839,
                          -2.09439510239, -3.05432619099])
        q_max = np.array([2.96705972839, 2.09439510239, 2.96705972839, 2.09439510239, 2.96705972839,
                          2.09439510239, 3.05432619099])
        q_dot_max = np.array([1.71, 1.74, 1.745, 2.269, 2.443, 3.142, 3.142])
        q_dot_min = -q_dot_max
        self.z_ground = -0.158
        self.robot_path = robot_path
        self.posture = posture
        self.box_position = [1.3, 0.07, -0.1586]

        self.robot = VelocityHedgehog(q_min, q_max, q_dot_min, q_dot_max, robot_path, model_exist=True)
        if filepath is not None:
            self.data = np.load(filepath, allow_pickle=True).item()
            self.timestamp = np.array(self.data['stamp'])
            self.actual_position = np.array(self.data['real_pos'])
            self.actual_velocity = np.array(self.data['real_vel'])
            self.target_position = np.array(self.data['target_pos'])
            self.target_velocity = np.array(self.data['target_vel'])
            self.obj_position = np.array(self.data['obj_trajectory'])

            self.actual_final_pos_ee, self.actual_final_J = self.robot.forward(self.actual_position[-1], posture=self.posture)
            self.nominal_throwing = []
            self.actual_throwing = []

            # release state of nominal and actual trajectory
            a_pos, J_act = self.robot.forward(self.actual_position[-1], posture=self.posture)
            actual_vel_ee = J_act @ self.actual_velocity[-1]

            t_pos, J_tar = self.robot.forward(self.target_position[-1], posture=self.posture)
            target_vel_ee = J_tar @ self.target_velocity[-1]

            self.nominal_throwing.append(t_pos)
            self.nominal_throwing.append(target_vel_ee[:3])
            self.actual_throwing.append(a_pos)
            self.actual_throwing.append(actual_vel_ee[:3])

            ground_index = np.argmax(self.obj_position[:, 2] < self.z_ground + 0.02)
            box_pos = self.obj_position[:ground_index]

            self.target_impact_position = np.array(self.box_position)
            self.real_impact_position = np.array(box_pos[-1])
            self.actual_trajectory  = self.compute_gravity_trajectory(a_pos,
                                                                    actual_vel_ee[:3],
                                                                    self.z_ground)
            self.actual_impact_position = self.actual_trajectory[-1]


        elif batch_path is not None:
            self.batch_path = batch_path
            self.all_files = [f for f in os.listdir(batch_path) if f.endswith('.npy')]
            self.results = []

    def plot_all_real_trajectories(self):
        plt.figure(figsize=(8, 6))

        landing_x = []
        landing_y = []

        for file in self.all_files:
            data = np.load(os.path.join(self.batch_path, file), allow_pickle=True).item()
            real_pos = np.array(data['obj_trajectory'])

            z_ground = self.z_ground
            first_below = np.argmax(real_pos[:, 2] < z_ground + 0.05)

            if first_below > 0:
                landing_point = real_pos[first_below - 1]
                landing_x.append(landing_point[0])
                landing_y.append(landing_point[1])

        if len(landing_x) > 0:
            plt.scatter(
                landing_x, landing_y,
                c='b', marker='o', alpha=0.6,
                label='Landing Points'
            )

            mean_x = np.mean(landing_x)
            mean_y = np.mean(landing_y)
            std_x = np.std(landing_x)
            std_y = np.std(landing_y)

            mae_x = np.mean(np.abs(np.array(landing_x) - mean_x))
            mae_y = np.mean(np.abs(np.array(landing_y) - mean_y))

            print(f'MAE (X): {mae_x:.4f} ± {std_x:.4f}')
            print(f'MAE (Y): {mae_y:.4f} ± {std_y:.4f}')

            plt.scatter(
                mean_x, mean_y,
                c='r', marker='o', s=200,
                label=f'Mean Point ({mean_x:.2f}, {mean_y:.2f})'
            )

            x_margin = 0.1
            y_margin = 0.1

            plt.xlim(mean_x - x_margin, mean_x + x_margin)
            plt.ylim(mean_y - y_margin, mean_y +y_margin)

            plt.xlabel('X (m)')
            plt.ylabel('Y (m)')
            plt.title(f'Landing Points Distribution (n={len(landing_x)})')
            plt.legend()
            plt.gca().set_aspect('equal', adjustable='box')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.show()
        else:
            print("No valid landing points found in the data.")

    def analyze_performance(self):
        metrics = []

        for file in self.all_files:
            filepath = os.path.join(self.batch_path, file)
            evaluator = TrackingEvaluation(self.robot_path, filepath=filepath)

            metric = self._calculate_single_metrics(evaluator)
            print(filepath, metric['release_pos_error'])
            metrics.append(metric)

            del evaluator

        df = pd.DataFrame(metrics)
        rmse_metrics = {
            'trajectory_pos_rmse': np.mean(df['trajectory_pos_rmse']),
            'trajectory_vel_rmse': np.mean(df['trajectory_vel_rmse']),
            'release_pos_rmse': np.mean(df['release_pos_error']),
            'real_target_rmse': np.mean(df['real_target_dist']),
            'real_actual_rmse': np.mean(df['real_actual_dist'])
        }

        max_abs_error_metrics = {
            # 'trajectory_pos_max_abs_error': np.max(np.abs(df['pos_error'])),
            # 'trajectory_vel_max_abs_error': np.max(np.abs(df['vel_error'])),
            'release_pos_max_abs_error': np.max(np.abs(df['release_pos_error'])),
            'real_target_max_abs_error': np.max(np.abs(df['real_target_dist'])),
            'real_actual_max_abs_error': np.max(np.abs(df['real_actual_dist']))
        }

        performance_metrics = {**rmse_metrics, **max_abs_error_metrics}
        performance_df = pd.DataFrame([performance_metrics])
        print("\n=== Performance RMSE Metrics ===")
        print(performance_df.to_string(index=False))

        return rmse_metrics

    def _calculate_single_metrics(self, evaluator):
        # the whole trajectory
        pos_error, vel_error, pos_rmse, vel_rmse = self._calculate_tracking_error(evaluator)

        release_pos_error = np.linalg.norm(
            evaluator.actual_throwing[0] -
            evaluator.obj_position[0]
        )

        real_target_dist = np.linalg.norm(evaluator.real_impact_position[:2] -
                                          evaluator.target_impact_position[:2])

        real_actual_dist = np.linalg.norm(evaluator.real_impact_position[:2] -
                                          evaluator.actual_impact_position[:2])

        return {
            'pos_error': pos_error,
            'vel_error': vel_error,
            'trajectory_pos_rmse': pos_rmse,
            'trajectory_vel_rmse': vel_rmse,
            'release_pos_error': release_pos_error,
            'real_target_dist': real_target_dist,
            'real_actual_dist': real_actual_dist
        }

    def _calculate_tracking_error(self, evaluator):
        actual_pos, target_pos = [], []
        actual_vel, target_vel = [], []

        for i in range(len(evaluator.timestamp)):
            a_pos, a_J = self.robot.forward(evaluator.actual_position[i], posture = self.posture)
            t_pos, t_J = self.robot.forward(evaluator.target_position[i], posture = self.posture)
            a_vel = a_J @ evaluator.actual_velocity[i]
            t_vel = t_J @ evaluator.target_velocity[i]

            actual_pos.append(a_pos)
            target_pos.append(t_pos)
            actual_vel.append(a_vel[:3])
            target_vel.append(t_vel[:3])
        actual_pos = np.array(actual_pos)
        target_pos = np.array(target_pos)
        actual_vel = np.array(actual_vel)
        target_vel = np.array(target_vel)

        pos_error = np.linalg.norm(actual_pos - target_pos, axis=1)
        pos_rmse = np.sqrt(np.mean(pos_error ** 2))

        vel_error = np.linalg.norm(actual_vel - target_vel, axis=1)
        vel_rmse = np.sqrt(np.mean(vel_error ** 2))

        return pos_error, vel_error, pos_rmse, vel_rmse

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

    def plot_ee_tracking(self, posture):
        """Plot end-effector trajectories in 3D space"""
        # Calculate trajectories
        actual_pos_ee, actual_vel_ee = [], []
        target_pos_ee, target_vel_ee = [], []

        for i in range(len(self.timestamp)):
            # Actual trajectory
            a_pos, J_act = self.robot.forward(self.actual_position[i], posture=self.posture)
            actual_pos_ee.append(a_pos)
            actual_vel_ee.append(J_act @ self.actual_velocity[i])

            # Target trajectory
            t_pos, J_tar = self.robot.forward(self.target_position[i], posture=self.posture)
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
                              target_pos_ee.max() - target_pos_ee.min()]).max() / 2
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

    def plot_obj_fly_trajectory(self):
        if len(self.obj_position) < 2:
            print("Not enough points to plot (need at least 2 points)")
            return

        actual_throw_pos = np.array(self.actual_throwing[0])
        actual_throw_vel = np.array(self.actual_throwing[1])
        real_pos = np.array(self.obj_position)
        z_ground = self.z_ground

        first_below_idx = np.argmax(real_pos[:, 2] < z_ground + 0.05)
        real_pos = real_pos[:first_below_idx]

        actual_traj = self.compute_gravity_trajectory(actual_throw_pos, actual_throw_vel, z_ground)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(real_pos[:, 0], real_pos[:, 1], real_pos[:, 2],
                'b-', lw=2, label='Measured Trajectory')
        ax.scatter(real_pos[0, 0], real_pos[0, 1], real_pos[0, 2],
                   c='lime', s=120, marker='*', label='Start Point')

        # ax.scatter(self.box_position[0], self.box_position[1], self.box_position[2],
        #            c='lime', s=120, marker='o', label='target position')
        #
        # ax.plot(actual_traj[:, 0], actual_traj[:, 1], actual_traj[:, 2],
        #         'r:', lw=1.5, label='Actual Simulation')

        xx, yy = np.meshgrid(
            np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 20),
            np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 20)
        )
        zz = np.full_like(xx, z_ground)
        ax.plot_surface(xx, yy, zz, alpha=0.3, color='gray')

        ax.set_zlim(z_ground - 0.1, np.nanmax(real_pos[:, 2]) + 0.5)
        ax.view_init(elev=25, azim=-45)
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
        ax.set_zlabel('Z (m)', fontsize=10)
        ax.set_title(f'3D Trajectory Analysis (Ground Z: {z_ground:.3f}m)', pad=15)

        leg = ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98),
                        frameon=True, framealpha=0.9)
        leg.get_frame().set_edgecolor('k')

        plt.tight_layout()
        plt.show()

    def compute_gravity_trajectory(self, pos0, vel0, ground_z, num_points=100):
        g = 9.81
        x0, y0, z0 = pos0
        vx0, vy0, vz0 = vel0

        a = -0.5 * g
        b = vz0
        c = z0 - ground_z
        discriminant = b ** 2 - 4 * a * c

        if discriminant >= 0:
            t1 = (-b + np.sqrt(discriminant)) / (2 * a)
            t2 = (-b - np.sqrt(discriminant)) / (2 * a)
            t_total = max(t1, t2)
            t_total = max(t_total, 0)
        else:
            t_total = (-vz0 / g) * 2 if vz0 != 0 else 5.0

        t = np.linspace(0, t_total, num_points)
        x = x0 + vx0 * t
        y = y0 + vy0 * t
        z = z0 + vz0 * t - 0.5 * g * t ** 2

        valid = np.where(z >= ground_z - 0.1)[0]
        if len(valid) == 0:
            return np.empty((0, 3))
        last_valid = valid[-1]
        return np.column_stack((x[:last_valid + 1], y[:last_valid + 1], z[:last_valid + 1]))

def plot_joint_tracking_data(file_path):
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
    filepath = f'../output/data/ee_tracking/throw_tracking_batch/throwing_20250420_113236.npy'
    batch_path = '../output/data/ee_tracking/throw_tracking_batch/'
    robot_path = '../description/iiwa7_allegro_throwing.xml'
    evaluator = TrackingEvaluation(batch_path=batch_path, robot_path=robot_path)

    # Plot joint space tracking
    # evaluator.plot_joint_tracking()

    # Plot Cartesian space tracking
    # evaluator.plot_ee_tracking(posture='posture1')

    # Plot object fly trajectory
    # evaluator.plot_obj_fly_trajectory()


    # batch evaluation
    evaluator.plot_all_real_trajectories()

    # metrics evaluation
    # evaluator.analyze_performance()