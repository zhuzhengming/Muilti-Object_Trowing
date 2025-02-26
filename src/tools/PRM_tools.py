# import JSDF

from JSDF import arm_hand_JSDF

import numpy as np
import torch
from tools.KNN_test import FaissKNeighbors
from tools.D_star_lite import DStar

from motion_planning.NN_model.nn_model_eval import NN_hand_obj
from motion_planning.NN_model.nn_model_eval import NN_SCA_pro


class PRM:
    def __init__(self, s_start: np.ndarray, s_goal: np.ndarray, n=2000, use_GPU=True, dof=None, link_nums=30,edge_sample_num=20, safety_margin=0.005):

        if dof is None:
            dof = list(range(23))
        self.device = "cuda:0" if (torch.cuda.is_available() and use_GPU) else "cpu"
        self.dtype = torch.float32

        self.dim = len(s_start)
        # assert self.dim in [7, 16, 23]
        self.n = n
        self.start = tuple(s_start)
        self.goal = tuple(s_goal)
        self.safety_margin = safety_margin

        self.jsdf = arm_hand_JSDF(generate_grid=False, use_GPU=use_GPU)
        self.iiwa_bounds = np.vstack([self.jsdf.arm_hand_robot.arm.joint_lower_bound,
                                      self.jsdf.arm_hand_robot.arm.joint_upper_bound])
        self.hand_bounds = np.vstack([self.jsdf.arm_hand_robot.hand.joint_lower_bound,
                                      self.jsdf.arm_hand_robot.hand.joint_upper_bound])
        self.bounds = np.concatenate([self.iiwa_bounds, self.hand_bounds], axis=1)
        self.bounds = self.bounds[:, dof]
        # if self.dim == 23:
        #     print("PRM for arm-hand system")
        #
        #     self.link_nums = 30
        # elif self.dim == 7:
        #     print("PRM for iiwa arm only")
        #     self.bounds = self.iiwa_bounds
        #     self.link_nums = 8
        # else:
        #     print("PRM for allegro hand only")
        #     self.bounds = self.hand_bounds
        self.link_nums = link_nums
        self.env = environments()
        self.dof = dof
        assert len(self.dof) == self.dim

        self.samples = None
        self.samples_near = None
        self.s1 = None
        self.s2 = None
        self.path = None
        self.edge_sample_num = edge_sample_num
        self.graph = graph(dim=self.dim)
        self.knn_nums = int(np.ceil(np.e * (1 + 1 / self.dim))) + 7
        self.knn = None
        self.d_star = None
        self.pairs = None

    def rerun_PRM(self, optimize=False) -> None:
        """
        run PRM multiple times incase it cannot find a path
        :return:
        """
        trial = 0
        while self.path is None:
            print("Trial", trial)
            self.run_PRM()
            trial += 1
        print('Path length:', len(self.path))
        if optimize and self.path[-1] == self.goal:
            print('PRM* by optimize the path')
            self.run_PRM(samples=np.vstack(self.path))

    def run_PRM(self, samples=None):
        if samples is None:
            self.sampling()
        else:
            self.samples = np.vstack(self.path)
        self.knn = FaissKNeighbors(k=self.knn_nums + 1)  # knn
        self.run_knn()  # update edges
        self.d_star = DStar(self.start, self.goal, self.graph, "euclidean")
        self.pairs = self.update_collision_states()

        self.update_graph()
        self.find_path()

    def find_path(self):
        result = self.d_star.ComputePath()
        if result:
            path = self.d_star.extract_path()
            self.path = path

    def sampling(self):
        """
        sample n points within the bounds
        :return:
        """
        samples = np.random.uniform(self.bounds[0, :], self.bounds[1, :], size=(self.n, self.dim))  # (n, dim)
        self.samples = np.concatenate(
            [np.array(self.start).reshape(1, -1), np.array(self.goal).reshape(1, -1), samples])

    def run_knn(self):
        self.knn.fit(self.samples)
        self.samples_near = self.knn.predict(self.samples)[:, 1:, :]  # remove the first one, which is itself
        self.s1 = list(map(tuple, self.samples))  # start of each edge
        self.s2 = [list(map(tuple, self.samples_near[i, :, :])) for i in
                   range(self.samples_near.shape[0])]  # end of each edge
        edges = dict(zip(self.s1, self.s2))  # {v1: [a1,a2,a3], v2; [...],...}
        self.graph.edges = edges

    def update_collision_states(self):
        start_p = np.repeat(self.samples, repeats=self.knn_nums, axis=0)
        edge_samples_ = np.linspace(start_p, np.vstack(self.samples_near),
                                    self.edge_sample_num, axis=1)  # (n*k, edge_sample_num, dim)
        edge_samples = np.vstack(edge_samples_)  # (n*k*edge_sample_num, dim)

        # if self.dim == 23:
        pairs = self.collision_check(edge_samples, self.env.x_obj)
        return pairs

    def update_graph(self):
        start_end_p = []
        for i in range(len(self.s2)):
            for j in range(self.knn_nums):
                start_end_p.append(self.s1[i] + self.s2[i][j])

        self.d_star.graph.E = dict(zip(start_end_p, self.pairs))

    def collision_check(self, q: np.ndarray, x: np.ndarray) -> np.ndarray:
        """

        :param q: (qn, 23), sample points in C space
        :param x: (xn, 3), obstacle point cloud
        :return:  (qn, 30) distances to each link
        """

        xn = x.shape[0]
        qn = q.shape[0]
        q = torch.from_numpy(q).type(self.dtype).to(self.device)
        x = torch.from_numpy(x).type(self.dtype).to(self.device)

        q, x = q.repeat_interleave(xn, 0), x.repeat(qn, 1)

        qx = torch.hstack((q, x))
        tmp = self.jsdf.calculate_signed_distance_raw_input(qx)
        tmp = tmp.reshape(qn, xn, self.link_nums)
        dis, _ = torch.min(tmp, 1)  # (qn, 30) # min distances for all obstacle to the arm-hand system
        dis_reduced, _ = torch.min(dis.reshape(-1, self.edge_sample_num, self.link_nums), 1)

        dis_reduced_min, _ = torch.min(dis_reduced, 1)  # ()
        # a = 1
        return (dis_reduced_min > self.safety_margin).cpu().numpy()


class collision_check:
    def __init__(self, x_obj, g=None, use_cuda=False, right_hand=True, visualize=[2]):
        if g is None:
            g = [0, 1, 2, 3, 4]
        self.nn = NN_hand_obj(g=g, path_prefix_suffix=[
            '/home/xiao/research/lasa/iiwa_allegro_sim/motion_planning/NN_model/models/single_', '01.pt'],
                              use_cuda=use_cuda,
                              right=right_hand)
        if right_hand:
            self.nn_SCA = NN_SCA_pro(
                path_check_point='/home/xiao/research/lasa/iiwa_allegro_sim/motion_planning/NN_model/models/model_new_01_N5.pt',
                use_cuda=use_cuda,
                right_hand=right_hand)
        else:
            self.nn_SCA = NN_SCA_pro(
                path_check_point='/home/xiao/research/lasa/iiwa_allegro_sim/motion_planning/NN_model/models/model_new_01_N5_left.pt',
                use_cuda=use_cuda,
                right_hand=right_hand)
        self.use_cuda = use_cuda
        self.g = g
        self.x_obj = x_obj
        self.x_obj_gpu = torch.Tensor(x_obj).to('cuda:0') if self.use_cuda else torch.Tensor(x_obj)
        self.q_gpu = []

        nums = 300
        self.nums = nums
        lb = self.nn.hand_bound[0, (visualize[0] - 1) * 4 + 1:visualize[0] * 4 - 1]
        ub = self.nn.hand_bound[1, (visualize[0] - 1) * 4 + 1:visualize[0] * 4 - 1]
        x_ = np.linspace(lb[0], ub[0], nums)
        y_ = np.linspace(lb[1], ub[1], nums)
        x_grid, y_grid = np.meshgrid(x_, y_, )
        self.x1 = x_grid.flatten().reshape(-1, 1)
        self.y1 = y_grid.flatten().reshape(-1, 1)
        # q_ = np.concatenate([np.zeros([nums ** 2, 5]), self.x1, self.y1, np.zeros([nums ** 2, 9])], axis=1)
        q_ = np.concatenate([np.zeros([nums ** 2, 1]), self.x1, self.y1, np.zeros([nums ** 2, 1])], axis=1)
        self.q_grid_gpu = torch.Tensor(q_).to('cuda:0') if use_cuda else torch.Tensor(q_)
        self.q_gpu = []
        self.x_grid = x_grid
        self.y_grid = y_grid

    def get_full_map_dis(self, q_now=None, x_obj_=None):
        if q_now is None:
            q_now = np.zeros(16)
            q_now[12] = 0
        q_all = np.concatenate([np.repeat(q_now[:9].reshape(1, -1), self.nums ** 2, axis=0), self.x1, self.y1,
                                np.repeat(q_now[11:].reshape(1, -1), self.nums ** 2, axis=0)], axis=1)
        q_all_gpu = torch.Tensor(q_all).to('cuda:0') if self.use_cuda else torch.Tensor(q_all)
        q_all_gpu = self.q_grid_gpu
        # input obstacles
        if x_obj_ is None:
            x_obj_ = self.x_obj_gpu
        if not isinstance(x_obj_, torch.Tensor):
            x_obj_ = torch.Tensor(x_obj_).to('cuda:0') if self.use_cuda else torch.Tensor(x_obj_)
        dis = self.nn.eval_multiple(q_all_gpu, x_obj_, real_dis=False, only_min_dis=True, gradient=False)
        return dis.reshape(self.nums, self.nums)

    def get_dis(self, q=None, gradient=False, x_obj=None, sample=None, real_dis=False, safety_dis=0, dx=False):
        if safety_dis != 0:
            real_dis = True
        if x_obj is not None:
            # x_obj_gpu = x_obj
            x_obj_gpu = torch.Tensor(x_obj).to('cuda:0') if self.use_cuda else torch.Tensor(x_obj)
        else:
            x_obj_gpu = self.x_obj_gpu

        # global x_obj_gpu
        if q is None:
            # this is for PRM samples
            output = self.nn.eval_multiple(self.q_gpu, x_obj_gpu, real_dis=real_dis, only_min_dis=True,
                                           gradient=False)
            if sample is None:
                return output
            else:
                output_bool = output > safety_dis
                output_bool = output_bool.reshape(-1, sample)
                pairs = np.all(output_bool, axis=1)
                return pairs
        else:
            if isinstance(q, tuple):
                q = np.array(q)
            if len(q.shape) == 1:
                q = q.reshape(1, -1)

            if q.shape == (1, 2) or q.shape == (1, 4):
                if q.shape == (1, 2):
                    q = np.array([0, q[0, 0], q[0, 1], 0])
                if gradient:
                    # output, grad = self.nn.eval(q, x_obj_gpu, real_dis=real_dis, only_min_dis=True,
                    #                             gradient=gradient)  # single q, multiple x
                    # return output[self.g[0]], grad[self.g[0]][1:3]
                    output, grad = self.nn.eval_multiple(torch.Tensor(q.reshape(1, -1)), x_obj_gpu, real_dis=real_dis,
                                                         only_min_dis=True,
                                                         gradient=gradient)  # single q, multiple x
                    if q.shape == (1, 2):
                        return output[0], grad[0, 1:3]
                    else:
                        return output[0], grad[0, :]
                else:
                    output = self.nn.eval(q, x_obj_gpu, real_dis=real_dis, only_min_dis=True, gradient=gradient)
                    return output[self.g[0]]
            elif q.shape[0] > 1:
                if q.shape[1] == 16:  # (n, 16)
                    pass
                    q = torch.Tensor(q).to('cuda:0') if self.use_cuda else torch.Tensor(q)
                    if gradient:
                        output, grad = self.nn.eval_multiple(q, x_obj_gpu, real_dis=False, only_min_dis=True,
                                                             gradient=gradient, dx=dx)  # multiple q, single x
                        return output, grad
                        # if dx:
                        #     return output, grad
                        # else:
                        #     return output, grad[:, :16]
                    else:
                        output = self.nn.eval_multiple(q, x_obj_gpu, real_dis=real_dis, only_min_dis=True,
                                                       gradient=False)  # multiple q,  multiple x
                        if sample is None:
                            # return np.min(output)
                            return output
                        else:
                            output_bool = output > safety_dis
                            output_bool = output_bool.reshape(-1, sample)
                            pairs = np.all(output_bool, axis=1)  # this way is 4 times faster than below
                            # pairs = [all(output[j * sample: (j + 1) * sample] > 0) for j in
                            #          range(int(n / sample))]
                            return pairs  # bool type for collision

                elif q.shape[1] == 2:  # nx2
                    n = q.shape[0]
                    q = np.concatenate([np.zeros([n, 1]), q, np.zeros([n, 1])], axis=1)
                    q = torch.Tensor(q).to('cuda:0') if self.use_cuda else torch.Tensor(q)
                    if gradient:
                        output, grad = self.nn.eval_multiple(q, x_obj_gpu, real_dis=real_dis, only_min_dis=True,
                                                             gradient=gradient, dx=dx)  # multiple q, single x
                        if dx:
                            return output, grad
                        else:
                            return output, grad[:, 1:3]
                    else:
                        output = self.nn.eval_multiple(q, x_obj_gpu, real_dis=real_dis, only_min_dis=True,
                                                       gradient=False)  # multiple q,  multiple x
                        if sample is None:
                            # return np.min(output)
                            return output
                        else:
                            output_bool = output > safety_dis
                            output_bool = output_bool.reshape(-1, sample)
                            pairs = np.all(output_bool, axis=1)  # this way is 4 times faster than below
                            # pairs = [all(output[j * sample: (j + 1) * sample] > 0) for j in
                            #          range(int(n / sample))]
                            return pairs  # bool type for collision
                else:
                    raise NotImplementedError
            elif q.shape == (1, 16):
                q = torch.Tensor(q).to('cuda:0') if self.use_cuda else torch.Tensor(q)
                if gradient:
                    output, grad = self.nn.eval_multiple(q, x_obj_gpu, real_dis=real_dis, only_min_dis=True,
                                                         gradient=gradient)  # single q, multiple x
                    return output, grad
                else:
                    output = self.nn.eval_multiple(q, x_obj_gpu, real_dis=real_dis, only_min_dis=True,
                                                   gradient=gradient)
                    return output

            else:
                raise ValueError('q has a wrong shape', q.shape)

    def obstacle_free(self, q):
        """
        Check if a location resides inside of an obstacle
        :param q: location to check
        :return: True if not inside an obstacle, False otherwise
        """
        # return self.obs.count(x) == 0
        return self.get_dis(q, x_obj=self.x_obj) > 0

    def SCA_eval(self, q=None, gradient=False, sample=None, safety_dis=0):
        if q is None:
            q = self.q_gpu
        real_dis = False
        if safety_dis != 0:
            real_dis = True
        if gradient:
            output, grad = self.nn_SCA.eval(q, gradient=gradient, real_dis=real_dis)
        else:
            output = self.nn_SCA.eval(q, gradient=gradient, real_dis=real_dis)

        if sample is None:
            # return np.min(output)
            if gradient:
                return output, grad
            else:
                return output
        else:
            output_bool = output > safety_dis
            output_bool = output_bool.reshape(-1, sample)
            pairs = np.all(output_bool, axis=1)
            return pairs

    def collision_hand_obj_SCA(self, q=None, sample=None, safety_dis=0):
        pair_1 = self.get_dis(q=q, sample=sample, safety_dis=safety_dis)
        pair_2 = self.SCA_eval(q=q, sample=sample, safety_dis=safety_dis)

        return np.logical_and(pair_1, pair_2)


class graph:
    def __init__(self, dim=2):
        self.dim = dim
        self.V_count = 0

        self.E = {}  # edges  {v : [v1, v2, v2,..., vn], v' :  } with collision-free edges
        self.edges = {}  # edges  {v : [v1, v2, v2,..., vn], v' :  } with collision-free edges
        self.V_dis_grad = {}  # store the distance and gradients for all vertices

    def nearby(self, q, k):
        #  return the near vertices by knn
        # k(n) := k_{PRM} * log(n) , k_{PRM} > e(1 + 1/dim)
        # return self.V.nearest(q, num_results=k, objects='raw')

        pass

    # def add_vertex(self, v):
    #     self.V.insert(0, v + v, v)
    #     self.V_count += 1


class environments:
    def __init__(self):
        # initilize the table

        # load possible objects
        # table 0.9 m in world frame
        x_obj = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 2]])
        self.x_obj = x_obj
        pass

    def update_point_cloud(self) -> np.ndarray:
        """
        update the current pointcloud as obstacles
        :return:
        """
        pass

    def static_env(self):
        pass


if __name__ == '__main__':
    start = np.zeros(23)
    goal = np.ones(23)
    PRM_test = PRM(start, goal)
    PRM_test.rerun_PRM(optimize=True)
    print(PRM_test.path)



    # jsdf_model = arm_hand_JSDF(generate_grid=False)
    # jsdf_model.arm
    # q = [0] * 23
    # obstacle_point_clouds = np.array([[1., 1., 1.], [2., 2., 2.]])
    #
    # # Without gradient
    # distances = jsdf_model.calculate_signed_distance(obstacle_point_clouds)
    # print(distances)
    #
    # # With gradient (carful distances dimentsion is [2, 30] or somthing with corresponds to [nb_points, nb_links]. Or maybe it is the transposed... to check)
    # (distances, gradient) = jsdf_model.whole_arm_inference_with_gradient(obstacle_point_clouds)
    # print(distances)
    # print(gradient)
