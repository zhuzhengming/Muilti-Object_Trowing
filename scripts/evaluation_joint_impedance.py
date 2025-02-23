import numpy as np
import matplotlib.pyplot as plt

def plot_arm_data():
    for i in range(7):
        filename = f'../output/sin_test_{i}.npy'
        data = np.load(filename, allow_pickle=True).item()

        timestamp = data['timestamp']








    plt.tight_layout()
    plt.show()

plot_arm_data()
