import numpy as np
from concurrent.futures import ProcessPoolExecutor
import os
import time
from pathlib import Path
from trajectory_generator import TrajectoryGenerator
from tqdm import tqdm
import shutil

class OptimalSolver:
    def __init__(self):
        # Define joint and box limits
        self.q_min = np.array([-2.96705972839, -2.09439510239, -2.96705972839, -2.09439510239, -2.96705972839,
                               -2.09439510239, -3.05432619099])
        self.q_max = np.array([2.96705972839, 2.09439510239, 2.96705972839, 2.09439510239, 2.96705972839,
                               2.09439510239, 3.05432619099])
        self.q_dot_max = np.array([1.71, 1.74, 1.745, 2.269, 2.443, 3.142, 3.142])
        self.q_dot_min = -self.q_dot_max

        self.box_min = np.array([-1.4, -1.4, -0.1])
        self.box_max = np.array([1.4, 1.4, 0.1])

        # Fixed q0
        self.q0 = np.array([-1.5783, 0.1498, 0.1635, -0.7926, -0.0098, 0.6, 1.2881])

        hedgehog_path = '../hedgehog_revised'
        brt_path = '../brt_data'
        xml_path = '../description/iiwa7_allegro_throwing.xml'
        self.trajectoryGenerator = TrajectoryGenerator(self.q_max, self.q_min,
                                                       hedgehog_path, brt_path,
                                                       xml_path, q0=self.q0, model_exist=True)

        self.X_min = np.concatenate([self.q_min, self.box_min, self.box_min])
        self.X_max = np.concatenate([self.q_max, self.box_max, self.box_max])
        self.Y_min = np.concatenate([self.q_min, self.q_dot_min, self.q_min, self.q_dot_min])
        self.Y_max = np.concatenate([self.q_max, self.q_dot_max, self.q_max, self.q_dot_max])

        self.plane_resolution = 0.05 * 2
        self.height_resolution = 0.02 * 2
        self.box_grid = [np.arange(box_min, box_max, resolution)
                         for box_min, box_max, resolution in zip(self.box_min, self.box_max,
                                                                 [self.plane_resolution, self.plane_resolution,
                                                                  self.height_resolution])]
    def sample_filter(self):
        valid_keys = []
        total_samples = 0
        valid_count = 0

        box_points = np.array(np.meshgrid(*self.box_grid)).T.reshape(-1, 3)

        with tqdm(total=len(box_points) ** 2, desc="Filtering samples") as pbar:
            for box_A in box_points:
                for box_B in box_points:
                    total_samples += 1

                    mag_A = np.linalg.norm(box_A[:2])
                    mag_B = np.linalg.norm(box_B[:2])

                    if 1.25 <= mag_A <= 1.4 and 1.25 <= mag_B <= 1.4:
                        valid_keys.append((box_A, box_B))
                        valid_count += 1

                    pbar.update(1)

        print(f"Valid samples ratio: {valid_count / total_samples:.2%}")
        return valid_keys

    def solve(self, box_A, box_B):
        box_positions = np.array([box_A, box_B])
        _, best_throw_config_pair, _ = (
            self.trajectoryGenerator.multi_waypoint_solve(box_positions, animate=False))

        if best_throw_config_pair is None:
            return None

        desire_q_A = best_throw_config_pair[0][0]
        desire_q_A_dot = best_throw_config_pair[0][3]
        desire_q_B = best_throw_config_pair[1][0]
        desire_q_B_dot = best_throw_config_pair[1][3]

        return np.array([desire_q_A, desire_q_A_dot, desire_q_B, desire_q_B_dot])

    def data_generation(self,
                        save_path_X="../training_data/training_X.npy",
                        save_path_Y="../training_data/training_Y.npy",
                        batch_size=10000):
        valid_pairs = self.sample_filter()
        temp_dir = Path(save_path_X).parent / "temp"
        temp_dir.mkdir(exist_ok=True, parents=True)

        progress_file = temp_dir / "progress.txt"
        start_idx = 0

        if progress_file.exists():
            with open(progress_file) as f:
                start_idx = int(f.read().strip()) + 1
            print(f"Resuming from index {start_idx}")

        batch_X, batch_Y = [], []
        batch_num = len(list(temp_dir.glob("temp_X_*.npy")))

        try:
            for idx, (box_A, box_B) in enumerate(tqdm(valid_pairs[start_idx:],
                                                      initial=start_idx,
                                                      total=len(valid_pairs))):
                solution = self.solve(box_A, box_B)
                if solution is not None:
                    batch_X.append(np.concatenate([box_A, box_B]))
                    batch_Y.append(solution.flatten())

                    if len(batch_X) >= batch_size:
                        self._save_batch(temp_dir, batch_num, batch_X, batch_Y)
                        batch_num += 1
                        batch_X.clear()
                        batch_Y.clear()

                if idx % 1000 == 0:
                    with open(progress_file, 'w') as f:
                        f.write(str(idx + start_idx))

            if batch_X:
                self._save_batch(temp_dir, batch_num, batch_X, batch_Y)
            self._merge_batches(temp_dir, save_path_X, save_path_Y)
            shutil.rmtree(temp_dir)
            if progress_file.exists():
                progress_file.unlink()

        except KeyboardInterrupt:
            print("\nsave process...")
            if batch_X:
                self._save_batch(temp_dir, batch_num, batch_X, batch_Y)
            with open(progress_file, 'w') as f:
                f.write(str(idx + start_idx))
            raise

        return np.load(save_path_X), np.load(save_path_Y)

    def _save_batch(self, temp_dir, batch_num, X, Y):
        X_array = np.array(X)
        Y_array = np.array(Y)
        np.save(temp_dir / f"temp_X_{batch_num:04d}.npy", X_array)
        np.save(temp_dir / f"temp_Y_{batch_num:04d}.npy", Y_array)

    def _merge_batches(self, temp_dir, save_path_X, save_path_Y):
        X_files = sorted(temp_dir.glob("temp_X_*.npy"),
                         key=lambda x: int(x.stem.split("_")[-1]))
        Y_files = sorted(temp_dir.glob("temp_Y_*.npy"),
                         key=lambda x: int(x.stem.split("_")[-1]))

        X_list = []
        Y_list = []

        for x_file, y_file in zip(X_files, Y_files):
            X_list.append(np.load(x_file))
            Y_list.append(np.load(y_file))

        X_array = np.concatenate(X_list, axis=0)
        Y_array = np.concatenate(Y_list, axis=0)

        np.save(save_path_X, X_array)
        np.save(save_path_Y, Y_array)

    def _count_samples(self, temp_dir):
        return sum(len(np.load(f)) for f in temp_dir.glob("temp_X_*.npy"))


if __name__ == "__main__":
    Solver = OptimalSolver()

    X, Y = Solver.data_generation(batch_size=10000)
