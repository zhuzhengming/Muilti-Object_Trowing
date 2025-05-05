from trajectory_generator import TrajectoryGenerator
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

class OptimalSolver:
    def __init__(self, num_samples=1000):
        hedgehog_path = '../hedgehog_revised'
        brt_path = '../brt_data'
        xml_path = '../description/iiwa7_allegro_throwing.xml'

        self.q_min = np.array([-2.96705972839, -2.09439510239, -2.96705972839, -2.09439510239, -2.96705972839,
                               -2.09439510239, -3.05432619099])
        self.q_max = np.array([2.96705972839, 2.09439510239, 2.96705972839, 2.09439510239, 2.96705972839,
                               2.09439510239, 3.05432619099])

        self.box_min = np.array([-1.5, -1.5, -0.3])
        self.box_max = np.array([1.5, 1.5, 0.3])
        self.num_samples = num_samples

    def solve(self, q0, box_A, box_B):
        hedgehog_path = '../hedgehog_revised'
        brt_path = '../brt_data'
        xml_path = '../description/iiwa7_allegro_throwing.xml'
        self.trajectoryGenerator = TrajectoryGenerator(self.q_max, self.q_min,
                                                       hedgehog_path, brt_path,
                                                       xml_path, q0=q0)

        box_positions = np.array([box_A, box_B])
        final_trajectory, best_throw_config_pair, intermediate_time = (
            self.trajectoryGenerator.multi_waypoint_solve(box_positions,animate=False))

        desire_q_A = best_throw_config_pair[0][0]
        desire_q_A_dot = best_throw_config_pair[0][3]
        desire_q_B = best_throw_config_pair[1][0]
        desire_q_B_dot = best_throw_config_pair[1][3]

        return np.array([desire_q_A, desire_q_A_dot, desire_q_B, desire_q_B_dot])

    def sampler(self):
        q0 = np.random.uniform(self.q_min, self.q_max, 7)
        box_A = np.random.uniform(self.box_min, self.box_max, 3)
        box_B = np.random.uniform(self.box_min, self.box_max, 3)
        return q0, box_A, box_B


class NeuralNetworkModel(nn.Module):
    def __init__(self):
        super(NeuralNetworkModel, self).__init__()
        self.fc1 = nn.Linear(13, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 28)
        self.num_samples = 1000

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

    def data_generation(self):
        X = []
        Y = []
        solver = OptimalSolver(self.num_samples)
        for _ in range(self.num_samples):
            q0, qA, qB = solver.sampler()
            optimal_solution = solver.solve(q0, qA, qB)
            if optimal_solution is not None:
                X.append(np.concatenate([q0, qA, qB]))
                Y.append(optimal_solution)
        X = np.array(X)
        y = np.array(y)

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

        return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

    def training(self, X_train_tensor, y_train_tensor, num_epochs=100):

        model = NeuralNetworkModel()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        num_epochs = 100
        for epoch in range(num_epochs):
            self.train()
            optimizer.zero_grad()

            predictions = model(X_train_tensor)

            loss = criterion(predictions, y_train_tensor)

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    def eveluation(self, model, X_test_tensor, y_test_tensor):
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_tensor)
            test_loss = criterion(y_pred, y_test_tensor)
            print(f"Test Loss: {test_loss.item():.4f}")



if __name__ == "__main__":
    num_samples = 1000
    model_class = NeuralNetworkModel()

    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = model_class.data_generation(num_samples)

    model = model_class.training(X_train_tensor, y_train_tensor, num_epochs=100)

    model_class.evaluation(model, X_test_tensor, y_test_tensor)