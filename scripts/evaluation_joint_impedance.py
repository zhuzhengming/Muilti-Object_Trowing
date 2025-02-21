import numpy as np
import matplotlib.pyplot as plt

def plot_arm_data():

    fig, axs = plt.subplots(7, 2, figsize=(12, 18))

    for i in range(7):

        filename = f'../output/sin_test_{i}.npy'
        data = np.load(filename, allow_pickle=True).item()

        timestamp = data['timestamp']
        position = data['position']
        error_percent = data['error']


        print(f"Joint {i} - Timestamp shape: {timestamp.shape}, Position shape: {position.shape}, Error shape: {error_percent.shape}")


        axs[i, 0].plot(timestamp, position, label=f'Actual Position (q) for Joint {i}', linestyle='-', color='blue')
        axs[i, 0].set_xlabel('Timestamp')
        axs[i, 0].set_ylabel('Position')
        axs[i, 0].set_title(f'Trajectory of Joint {i}')
        axs[i, 0].grid(True)


        max_pos_idx = np.argmax(position)
        min_pos_idx = np.argmin(position)
        axs[i, 0].annotate(f'Max Position: {position[max_pos_idx]:.2f}',
                           xy=(timestamp[max_pos_idx], position[max_pos_idx]),
                           xytext=(timestamp[max_pos_idx] + 1, position[max_pos_idx] + 0.05),
                           arrowprops=dict(facecolor='green', arrowstyle="->"))
        axs[i, 0].annotate(f'Min Position: {position[min_pos_idx]:.2f}',
                           xy=(timestamp[min_pos_idx], position[min_pos_idx]),
                           xytext=(timestamp[min_pos_idx] + 1, position[min_pos_idx] - 0.05),
                           arrowprops=dict(facecolor='red', arrowstyle="->"))


        axs[i, 1].plot(timestamp, error_percent, label=f'Position Error for Joint {i}', linestyle='-', color='red')
        axs[i, 1].set_xlabel('Time (s)')
        axs[i, 1].set_ylabel('Error (%)')
        axs[i, 1].set_title(f'Position Error for Joint {i}')
        axs[i, 1].grid(True)


        max_error_idx = np.argmax(error_percent)
        min_error_idx = np.argmin(error_percent)
        axs[i, 1].annotate(f'Max Error: {error_percent[max_error_idx]:.2f}%',
                           xy=(timestamp[max_error_idx], error_percent[max_error_idx]),
                           xytext=(timestamp[max_error_idx] + 1, error_percent[max_error_idx] + 2),
                           arrowprops=dict(facecolor='blue', arrowstyle="->"))
        axs[i, 1].annotate(f'Min Error: {error_percent[min_error_idx]:.2f}%',
                           xy=(timestamp[min_error_idx], error_percent[min_error_idx]),
                           xytext=(timestamp[min_error_idx] + 1, error_percent[min_error_idx] - 2),
                           arrowprops=dict(facecolor='orange', arrowstyle="->"))

    plt.tight_layout()
    plt.show()


plot_arm_data()
