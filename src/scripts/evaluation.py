import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

def plot_arm_data():
    # Create figure with 7 rows and 2 columns for position and error plots, and 7 rows and 2 columns for velocity and error plots
    fig1, axs1 = plt.subplots(7, 2, figsize=(12, 18))  # Adjust the figure size to suit the layout
    filename = f'../output/data/throwing.npy'
    data = np.load(filename, allow_pickle=True).item()

    error_pos_array = data['error_pos_array']
    error_vel_array = data['error_vel_array']
    joint_velo_array = data['joint_velo_array']

    # Plotting the position errors
    for i in range(7):  # Assuming 7 joints
        axs1[i, 0].plot(error_pos_array[:, i], label=f'Joint {i+1} Error')
        axs1[i, 0].set_title(f'Joint {i+1} Position Error')
        axs1[i, 0].set_xlabel('Time')
        axs1[i, 0].set_ylabel('Position Error')
        axs1[i, 0].legend()

    # Plotting the velocity errors
    for i in range(7):
        axs1[i, 1].plot(error_vel_array[:, i], label=f'Joint {i+1} Error')
        axs1[i, 1].set_title(f'Joint {i+1} Velocity Error')
        axs1[i, 1].set_xlabel('Time')
        axs1[i, 1].set_ylabel('Velocity Error')
        axs1[i, 1].legend()

    # Create a new figure for joint velocities
    fig2, axs2 = plt.subplots(7, 1, figsize=(12, 18))  # One plot per joint for velocities

    for i in range(7):  # Assuming 7 joints
        axs2[i].plot(joint_velo_array[:, i], label=f'Joint {i+1} Velocity')
        axs2[i].set_title(f'Joint {i+1} Velocity')
        axs2[i].set_xlabel('Time')
        axs2[i].set_ylabel('Velocity')
        axs2[i].legend()

    # Adjust layout to make sure everything fits
    plt.tight_layout()
    plt.show()

# Call the function to plot the data
plot_arm_data()
