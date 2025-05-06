import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
from trajectory_generator import TrajectoryGenerator

class OptimalSolver:
    def __init__(self):
        hedgehog_path = '../hedgehog_revised'
        brt_path = '../brt_data'
        xml_path = '../description/iiwa7_allegro_throwing.xml'

        self.q_min = np.array([-2.96705972839, -2.09439510239, -2.96705972839, -2.09439510239, -2.96705972839,
                               -2.09439510239, -3.05432619099])
        self.q_max = np.array([2.96705972839, 2.09439510239, 2.96705972839, 2.09439510239, 2.96705972839,
                               2.09439510239, 3.05432619099])

        self.box_min = np.array([-1.4, -1.4, -0.3])
        self.box_max = np.array([1.4, 1.4, 0.3])

    def solve(self, q0, box_A, box_B):
        hedgehog_path = '../hedgehog_revised'
        brt_path = '../brt_data'
        xml_path = '../description/iiwa7_allegro_throwing.xml'
        self.trajectoryGenerator = TrajectoryGenerator(self.q_max, self.q_min,
                                                       hedgehog_path, brt_path,
                                                       xml_path, q0=q0, model_exist=True)

        box_positions = np.array([box_A, box_B])
        final_trajectory, best_throw_config_pair, intermediate_time = (
            self.trajectoryGenerator.multi_waypoint_solve(box_positions, animate=False))

        if best_throw_config_pair is None:
            return None

        desire_q_A = best_throw_config_pair[0][0]
        desire_q_A_dot = best_throw_config_pair[0][3]
        desire_q_B = best_throw_config_pair[1][0]
        desire_q_B_dot = best_throw_config_pair[1][3]

        return np.array([desire_q_A, desire_q_A_dot, desire_q_B, desire_q_B_dot])

    def sampler(self):
        while True:
            q0 = np.random.uniform(self.q_min, self.q_max, 7)
            box_A = np.random.uniform(self.box_min, self.box_max, 3)
            box_B = np.random.uniform(self.box_min, self.box_max, 3)

            if (self.box_min[0] < box_A[0] < -1.0 and 1.0 < box_A[1] < 1.4 and -0.3 < box_A[2] < 0.3 and
                    self.box_min[0] < box_B[0] < -1.0 and 1.0 < box_B[1] < 1.4 and -0.3 < box_B[2] < 0.3):
                return q0, box_A, box_B


class NeuralNetworkModel(nn.Module):
    def __init__(self, num_samples=50000):
        super(NeuralNetworkModel, self).__init__()
        self.fc1 = nn.Linear(13, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 28)
        self.dropout = nn.Dropout(0.5)
        self.batch_norm = nn.BatchNorm1d(128)
        self.num_samples = num_samples

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.batch_norm(x)
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        return self.fc4(x)

    def data_generation(self,
                        save_path_X="../training_data/training_X.npy",
                        save_path_Y="../training_data/training_Y.npy"):
        X = []
        Y = []
        solver = OptimalSolver()

        for _ in tqdm(range(self.num_samples), desc="Generating Samples", unit="sample"):
            q0, box_A, box_B = solver.sampler()
            optimal_solution = solver.solve(q0, box_A, box_B)
            if optimal_solution is not None:
                q0 = np.array(q0).flatten()
                box_A = np.array(box_A).flatten()
                box_B = np.array(box_B).flatten()

                X.append(np.concatenate([q0, box_A, box_B]))
                Y.append(np.array(optimal_solution).flatten())

        X = np.array(X)
        Y = np.array(Y)

        np.save(save_path_X, X)
        np.save(save_path_Y, Y)

        print("data generation done. saved to file.")
        print(X.shape, Y.shape)
        return X, Y

    def train_model(self, X_train_tensor, y_train_tensor, num_epochs=100):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trained_model = NeuralNetworkModel()
        trained_model.to(device)
        X_train_tensor = X_train_tensor.to(device)
        y_train_tensor = y_train_tensor.to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(trained_model.parameters(), lr=0.001)

        for epoch in tqdm(range(num_epochs), desc="Training Epochs", unit="epoch"):
            trained_model.train()
            optimizer.zero_grad()

            predictions = trained_model(X_train_tensor)

            loss = criterion(predictions, y_train_tensor)

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

        torch.save(trained_model.state_dict(), "../NN_model/model.pth")
        return trained_model

    def evaluation(self, model, X_test_tensor, y_test_tensor):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        model.to(device)
        X_test_tensor = X_test_tensor.to(device)
        y_test_tensor = y_test_tensor.to(device)
        criterion = nn.MSELoss()
        with torch.no_grad():
            y_pred = model(X_test_tensor)
            test_loss = criterion(y_pred, y_test_tensor)
            print(f"Test Loss: {test_loss.item():.4f}")


if __name__ == "__main__":
    num_samples = 30000
    model_class = NeuralNetworkModel(num_samples)

    data_file = "../training_data/"
    X_path = data_file + "training_X.npy"
    Y_path = data_file + "training_Y.npy"
    if os.path.exists(X_path) or os.path.exists(Y_path):
        X_path = data_file + "training_X.npy"
        Y_path = data_file + "training_Y.npy"
        X = np.load(X_path, allow_pickle=True)
        Y = np.load(Y_path, allow_pickle=True)

        print("Loaded data from file.")
    else:
        print("Generating data...")
        X, Y = model_class.data_generation()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)

    model = model_class.train_model(X_train_tensor, y_train_tensor, num_epochs=1000)

    model_class.evaluation(model, X_test_tensor, y_test_tensor)
