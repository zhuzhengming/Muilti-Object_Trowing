import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

def plot_arm_data():
    # Create figure with 7 rows and 2 columns for position and error plots, and 7 rows and 2 columns for velocity and error plots
    fig1, axs1 = plt.subplots(7, 2, figsize=(12, 18))  # Adjust the figure size to suit the layout
    fig2, axs2 = plt.subplots(7, 2, figsize=(12, 18))
    for i in range(7):
        filename = f'../output/data/sin_test_{i}.npy'
        data = np.load(filename, allow_pickle=True).item()

        timestamp = data['timestamp']
        actual_position = data['actual_position']
        target_position = data['target_position']
        error_position_percent = data['error_position_percent']
        actual_velocity = data['actual_velocity']
        target_velocity = data['target_velocity']
        error_velocity_percent = data['error_velocity_percent']

        print(f"Joint {i} - Timestamp shape: {timestamp.shape}, Position shape: {actual_position.shape}, Error shape: {error_position_percent.shape}")

        # Plot the actual position and target position for joint i
        axs1[i, 0].plot(timestamp, actual_position, label=f'Actual Position (q) for Joint {i}', linestyle='-', color='blue')
        axs1[i, 0].plot(timestamp, target_position, label=f'Target Position for Joint {i}', linestyle='--', color='green')  # Target position curve
        axs1[i, 0].set_ylabel('Position')
        axs1[i, 0].set_title(f'Trajectory of Joint {i} with gravity compensation')
        axs1[i, 0].legend()
        axs1[i, 0].grid(True)

        # Plot the position error for joint i
        axs1[i, 1].plot(timestamp, error_position_percent, label=f'Position Error for Joint {i}', linestyle='-', color='red')
        axs1[i, 1].set_ylabel('Error (%)')
        axs1[i, 1].set_title(f'Position Error for Joint {i}')
        axs1[i, 1].grid(True)

        # Plot the actual velocity for joint i
        axs2[i, 0].plot(timestamp, actual_velocity, label=f'Actual Velocity (qd) for Joint {i}', linestyle='-', color='green')
        axs2[i, 0].plot(timestamp, target_velocity, label=f'Target Velocity for Joint {i}', linestyle='--', color='purple')  # Target velocity curve
        axs2[i, 0].set_ylabel('Velocity')
        axs2[i, 0].set_title(f'Actual Velocity of Joint {i} with gravity compensation')
        axs2[i, 0].legend()
        axs2[i, 0].grid(True)

        # Plot the velocity error for joint i
        axs2[i, 1].plot(timestamp, error_velocity_percent, label=f'Velocity Error for Joint {i}', linestyle='-', color='orange')
        axs2[i, 1].set_ylabel('Error (%)')
        axs2[i, 1].set_title(f'Velocity Error for Joint {i}')
        axs2[i, 1].grid(True)

    # Add x-axis labels for the last row of subplots
    axs1[-1, 0].set_xlabel('Timestamp')
    axs1[-1, 1].set_xlabel('Timestamp')
    axs2[-1, 0].set_xlabel('Timestamp')
    axs2[-1, 1].set_xlabel('Timestamp')

    plt.tight_layout()
    plt.show()

plot_arm_data()