import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

def plot_arm_data(joint_index):
    # Create figure with 3 subplots (Position, Velocity, Effort) for the third joint
    fig, axs = plt.subplots(3, 1, figsize=(12, 18))  # 3 rows for position, velocity, and effort plots

    filename = f'../output/data/throwing.npy'
    data = np.load(filename, allow_pickle=True).item()

    timestamp = np.array(data['stamp'])
    actual_position = np.array(data['real_pos'])
    actual_velocity = np.array(data['real_vel'])
    actual_effort = np.array(data['real_eff'])
    target_position = np.array(data['target_pos'])
    target_velocity = np.array(data['target_vel'])
    target_effort = np.array(data['target_eff'])

    joint_index = joint_index # Third joint

    # Plot Position data for the third joint
    axs[0].plot( actual_position[:, joint_index], label=f'Actual Position for Joint {joint_index}', linestyle='-', color='blue')
    axs[0].plot(target_position[:, joint_index], label=f'Target Position for Joint {joint_index}', linestyle='--', color='green')
    axs[0].set_ylabel('Position')
    axs[0].set_title(f'Joint {joint_index} Position Tracking')
    axs[0].grid(True)
    axs[0].legend(loc='best')

    # Plot Velocity data for the third joint
    axs[1].plot( actual_velocity[:, joint_index], label=f'Actual Velocity for Joint {joint_index}', linestyle='-', color='blue')
    axs[1].plot(target_velocity[:, joint_index], label=f'Target Velocity for Joint {joint_index}', linestyle='--', color='green')
    axs[1].set_ylabel('Velocity')
    axs[1].set_title(f'Joint {joint_index} Velocity Tracking')
    axs[1].grid(True)
    axs[1].legend(loc='best')

    # Plot Effort data for the third joint
    axs[2].plot(actual_effort[:, joint_index], label=f'Actual Effort for Joint {joint_index}', linestyle='-', color='blue')
    axs[2].plot(target_effort[:, joint_index], label=f'Target Effort for Joint {joint_index}', linestyle='--', color='green')
    axs[2].set_ylabel('Effort (Torque)')
    axs[2].set_title(f'Joint {joint_index} Effort Tracking')
    axs[2].grid(True)
    axs[2].legend(loc='best')

    # Add x-axis label for all subplots
    axs[2].set_xlabel('Timestamp')

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
    # plot_tracking_data(file_path = "../output/data/test_kp_kd/kp_kd_joint2.npy")
    for i in range(7):
        plot_arm_data(joint_index=i)
