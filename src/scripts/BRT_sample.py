"""
input: landing set, [0,0, r_dot0, z_dot0]
sampling and solving ODE problem

output:
- brt_data:[N, 4] - [r, z, r_dot, z_dot]
- brt_tensor: (num_zs, num_dis, num_phi, num_gamma, 5) - x = [r, z, r_dot, z_dot, v]
- brt_zs
"""


import sys
sys.path.append("../")
import time
import numpy as np
from scipy.integrate import odeint
from sys import getsizeof
from bisect import bisect_left

def flying_dynamics(t, s):
    g = 9.81
    dr = s[2]
    dz = s[3]
    dr_dot = 0.0
    dz_dot = - g
    return [dr, dz, dr_dot, dz_dot]


class BRT:
    def __init__(self, r_dot0, z_dot0, prefix=None):
        self.r_dot0 = r_dot0
        self.z_dot0 = z_dot0
        self.brt_path = (prefix if prefix else "") + 'brt_data'


    def BRT_generation(self):
        start = time.time()
        t = np.linspace(2.0, 0.0, 51)
        n_steps = t.shape[0]

        n_r_dot0 = self.r_dot0.shape[0]
        n_z_dot0 = self.z_dot0.shape[0]
        n_velo = n_r_dot0 * n_z_dot0

        [r_dot0s, z_dot0s] = np.meshgrid(self.r_dot0, self.z_dot0)

        r_dot0s_flat = r_dot0s.flatten()
        z_dot0s_flat = z_dot0s.flatten()

        brt_data = np.zeros((n_velo, n_steps, 4))

        for i in range(n_velo):
            # backward via ODE and flying dynamics
            sol = odeint(flying_dynamics,
                         [0,0, r_dot0s_flat[i], z_dot0s_flat[i]],
                         t, tfirst=True)
            brt_data[i, :, :] = sol

        print(brt_data[0,:])
        brt_data = brt_data.reshape(-1, 4)
        print("Original size: ", brt_data.shape[0])

        # filter out data with insane number
        brt_data = brt_data[(brt_data[:, 0] > -10)
                            & (brt_data[:, 1] > -5.0)
                            & (brt_data[:, 1] < 5.0)
                            & (brt_data[:, 2] < 10.0)
                            & (brt_data[:, 3] < 10.0)]
        print("Filtered size: ", brt_data.shape[0])

        if self.brt_path is not None:
            np.save(self.brt_path, brt_data)
        print("Generated", n_velo, "flying trajectories in %.3f" % (time.time() - start), "seconds with",
                  round(getsizeof(brt_data) / 1024 / 1024, 2), "MB")
        return brt_data

    def convert2tensor(self):

        self.brt_data = self.BRT_generation()

        step_robot_zs = 0.05
        robot_zs = np.arange(start=0.0, stop=1.10+0.01, step=step_robot_zs)
        robot_gamma = np.arange(start=20.0/180.0*np.pi, stop=70.0/180.0*np.pi, step=5.0/180.0*np.pi)

        bzstart = min(robot_zs) - step_robot_zs * np.ceil((min(robot_zs) - min(self.brt_data[:, 1])) / step_robot_zs)
        brt_zs = np.arange(start=bzstart, stop=max(self.brt_data[:, 1]) + 0.01, step=step_robot_zs)
        num_zs = brt_zs.shape[0]
        num_gammas = len(robot_gamma)  # brt_chunk.shape[1]

        brt_chunk = [[[] for j in range(num_gammas)] for i in range(num_zs)]
        states_num = 0
        for x in self.brt_data:
            # calculate z, gamma, v
            z = x[1]
            gamma = np.arctan2(x[3], x[2])
            # filter some states
            if gamma < min(robot_gamma) or gamma > max(robot_gamma):
                continue
            v = np.sqrt(x[2]**2 + x[3]**2)
            z_idx = self.insert_idx(brt_zs, z)
            ga_idx = self.insert_idx(robot_gamma, gamma)
            brt_chunk[z_idx][ga_idx].append(list(x) + [v])
            states_num += 1

        # delete empty chunks
        removes_i = 0
        while True:
            chunk = brt_chunk[removes_i]
            empty = True
            for j in range(num_gammas):
                if len(chunk[j]) > 0:
                    empty = False
                    break
            if not empty:
                break
            removes_i += 1
        brt_chunk = brt_chunk[removes_i:]
        brt_zs = brt_zs[removes_i:, ...]
        num_zs -= removes_i

        brt_tensor = []
        l=0
        while True:
            new_layer_brt = np.ones((num_zs, num_gammas, 5))
            stillhasvalue = False
            for i in range(num_zs):
                for j in range(num_gammas):
                    if len(brt_chunk[i][j]) < l+1:
                        new_layer_brt[i, j, :] = np.nan
                    else:
                        stillhasvalue = True
                        new_layer_brt[i, j, :] = brt_chunk[i][j][l]
            if not stillhasvalue:
                break
            brt_tensor.append(new_layer_brt)
            l += 1
        brt_tensor = np.array(brt_tensor)
        brt_tensor = np.moveaxis(brt_tensor, 0, 2)
        brt_tensor = np.expand_dims(brt_tensor, axis=(1, 2))
        print("Tensor Size: {0} with {1} states( occupation rate {2:0.1f}%)".format(
            brt_tensor.shape, states_num, 100 * states_num * 5.0 / (np.prod(brt_tensor.shape))))

        np.save(prefix + "/brt_tensor.npy", brt_tensor)
        np.save(prefix+ "/brt_zs.npy", brt_zs)

    def insert_idx(self, a, x):
        """
       :param a: sorted array, ascending
       :param x: element
       :return: the idx of the closest value to x
       """
        idx = bisect_left(a, x)
        if idx == 0:
            return idx
        elif idx == len(a):
            return idx - 1
        else:
            if (x - a[idx - 1]) < (a[idx] - x):
                return idx - 1
            else:
                return idx

if __name__ == "__main__":
    prefix = "../brt_data/"
    r_dot0 = np.arange(0.2, 2.0, 0.5)
    z_dot0 = np.arange(-5.0, -2, 0.5)
    BRT_generator = BRT(r_dot0, z_dot0, prefix=prefix)
    BRT_generator.convert2tensor()

