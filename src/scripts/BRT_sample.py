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
        t = np.linspace(2.0, 0.0, 51) # backwards
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
            # flying state for every timestamp
            brt_data[i, :, :] = sol

        brt_data = brt_data.reshape(-1, 4)

        # filter out data with insane number
        brt_data = brt_data[(brt_data[:, 0] > -10)
                            & (brt_data[:, 1] > -5.0)
                            & (brt_data[:, 1] < 5.0)
                            & (brt_data[:, 2] < 10.0)
                            & (brt_data[:, 3] < 10.0)]

        if self.brt_path is not None:
            np.save(self.brt_path, brt_data)

        return brt_data

    def convert2tensor(self):
        # generate original brt data
        self.brt_data = self.BRT_generation()
        GAMMA_TOLERANCE = 0.2 / 180.0 * np.pi
        Z_TOLERANCE = 0.01

        zs_step = 0.05
        delta_gamma = np.pi / 36
        gamma_offset = np.pi / 9
        robot_zs = np.arange(0, 1.2, zs_step)

        robot_gamma = np.arange(gamma_offset, np.pi / 2 - gamma_offset, delta_gamma)

        bzstart = min(robot_zs) - zs_step * np.ceil((min(robot_zs) - min(self.brt_data[:, 1])) / zs_step)
        brt_zs = np.arange(start=bzstart, stop=max(self.brt_data[:, 1]) + 0.01, step=zs_step)
        num_zs = brt_zs.shape[0]
        num_gammas = len(robot_gamma)  # brt_chunk.shape[1]

        brt_chunk = [[[] for j in range(num_gammas)] for i in range(num_zs)]
        states_num = 0
        pad_gamma = np.r_[-np.inf, robot_gamma]
        pad_zs = np.r_[-np.inf, brt_zs]
        # x = [r, z, r_dot, z_dot]
        for x in self.brt_data:
            z = x[1]
            gamma = np.arctan2(x[3], x[2])
            # drop some states
            # consider the maximum velocity robot can archive
            if gamma < min(robot_gamma) or gamma > max(robot_gamma):
                continue
            v = np.sqrt(x[2] ** 2 + x[3] ** 2)
            # if v > max: continue
            # argmax will be faster than where
            gi = np.argmax(abs(pad_gamma - gamma) < GAMMA_TOLERANCE)
            if gi == 0: continue
            zi = np.argmax(abs(pad_zs - z) < Z_TOLERANCE)
            if zi == 0: continue
            brt_chunk[zi - 1][gi - 1].append(list(x) + [v])
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
                    if len(brt_chunk[i][j]) < l + 1:
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
        # expend tensor for (dis, phi)

        # (z, dis, phi, gamma, 5):(r, z, r_dot, z_dot, v)
        brt_tensor = np.expand_dims(brt_tensor, axis=(1, 2))

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
    r_dot0 = np.arange(0.1, 3.0, 0.5)
    z_dot0 = np.arange(-5.0, -0.1, 0.5)
    BRT_generator = BRT(r_dot0, z_dot0, prefix=prefix)
    BRT_generator.convert2tensor()
    print("Done")

