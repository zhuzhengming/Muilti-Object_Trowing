"""
 An symbolic experssion of kinematic model for the allegro hand

"""
import sympy as sy
import xml.etree.ElementTree as ET
import numpy as np

import sys

sys.path.append("..")
import tools.rotations as rot
import os.path
import pickle


class Robot:
    def __init__(self, right_hand=True, path_prefix=''):
        if right_hand:
            path = path_prefix + 'description/allegro_all/allegro_right_bodies.xml'
            file_name = path_prefix + 'kinematics/q2pose_right.txt'
        else:
            path = path_prefix + 'description/allegro_all/allegro_left_bodies.xml'
            file_name = path_prefix + 'kinematics/q2pose_left.txt'
        tree = ET.parse(path)
        self.root = tree.getroot()

        # Jacobian calculation
        self.q_list = ['q_i', 'q_m', 'q_r', 'q_t']  # index, middle, ring, thumb
        self.T_list = None  # a list of symbolic expression for the fingertip poses based on joints
        if os.path.isfile(file_name):
            with open(file_name, 'rb') as f:
                self.T_list = pickle.load(f)
            print('Kinematic model has been loaded from ' + file_name)
        else:
            print(
                'Start to load xml file to build the kinematics. This might take about 20s, but only for the first time.')
            self.read_xml()  # read the kinematic chains from .xml file
            with open(file_name, 'wb') as f:
                pickle.dump(self.T_list, f)
            print('Kinematic model has been saved to ' + file_name)

        self.qi = sy.symbols('q_i:4')
        self.qm = sy.symbols('q_m:4')
        self.qr = sy.symbols('q_r:4')
        self.qt = sy.symbols('q_t:4')
        self.q = [self.qi, self.qm, self.qr, self.qt]

        # a list of lambda function for the fingertip poses, which will receive joints for numeric computation
        self.fk = [sy.lambdify([self.q[i]], self.T_list[i]) for i in range(4)]

        # for position jacobian 
        self.jac_syms = [sy.Matrix(self.T_list[i][:3, 3]).jacobian(list(self.q[i])) for i in
                         range(4)]  # symbolic value for position jacobian
        # a list of lambda function for the fingertip jacobians, which will receive joints for numeric computation
        self.jac = [sy.lambdify([self.q[i]], self.jac_syms[i]) for i in range(4)]

        lb = np.array([-0.59471316618668479, -0.29691276729768068, -0.27401187224153672, -0.32753605719833834] * 3 + [
            0.3635738998060688, -0.20504289759570773, -0.28972295140796106, -0.26220637207693537])
        ub = np.array([0.57181227113054078, 1.7367399715833842, 1.8098808147084331, 1.71854352396125431] * 3 + [
            1.4968131524486665, 1.2630997544532125, 1.7440185506322363, 1.8199110516903878])
        self.bounds = np.vstack([lb, ub])

    def read_xml(self):
        T_list = []  # the list of fingertip poses in (4, 4)
        site = []
        for body in self.root.iter('site'):
            site.append(body.attrib)

        for a in range(4):
            b = []
            j = []
            for body in self.root[a + 2].iter('body'):
                b.append(body.attrib)
            for body in self.root[a + 2].iter('joint'):
                j.append(body.attrib)

            T = sy.eye(4)
            num = len(j)
            q = sy.symbols(self.q_list[a] + ':' + str(num))

            # for i in range(len(b)):
            for i in range(4):
                pos = np.fromstring(b[i]['pos'], sep=' ')
                quat = np.fromstring(b[i]['quat'], sep=' ') if 'quat' in b[i] else np.array([1, 0, 0, 0.])
                # print(pos, quat)
                T = T * rot.pose2T(np.concatenate([pos, quat]))
                # print(T)
                T = T * rotation(q[i], j[i]['axis'])
                # print(T)

            s = np.fromstring(site[a]['pos'], sep=' ')
            Ts = sy.eye(4)
            Ts[:3, 3] = s
            T = T * Ts
            T = sy.simplify(T)
            T_list.append(T)
        self.T_list = T_list

    def forward_kine(self, q, quat=True):
        """
        forward kinematics for all fingers
        :param quat: return quaternion or rotation matrix
        :param q: numpy array  (16,) or (8,)
        :return: x:  poses for all fingertips
        """
        assert len(q) == 16

        poses = []
        for i in range(4):
            pose = self.fk[i](q[i * 4: 4 + i * 4])  # (4,4)
            if quat:
                pose = rot.T2pose(pose)  # (7, )
            poses.append(pose)
        return poses

    def get_jac_bad(self, q):
        """
         get the position jacobian for all fingertips

         !!!!!!!!!!!!!warning, this would be too slow if do the subs online
         please use the lambdify function version

        """
        subs_dic = subs_value([self.qi, self.qm, self.qr, self.qt], [q[:4], q[4:8], q[8:12], q[12:]])
        jac_list = []
        for i in range(4):
            jac_tmp = self.jac_syms[i].subs(subs_dic)  # numeric value
            jac_list.append(np.array(jac_tmp))
        return jac_list

    def get_jac(self, q):
        """
         get the position jacobian for all fingertips

        """
        jac_list = []
        for i in range(4):
            jac_tmp = self.jac[i](q[i * 4: 4 + i * 4])  # numeric value
            jac_list.append(np.array(jac_tmp))
        return jac_list


def rotation(theta, axis):
    """
    Given the rotation axis and angle, calculate the transformation matrix.
    :param theta:
    :param axis: (3, )
    :return:
    """
    axis = np.fromstring(axis, dtype=np.int8, sep=' ')
    T = sy.eye(4)
    tmp = np.sum(axis)
    c1 = sy.cos(theta * tmp)
    s1 = sy.sin(theta * tmp)
    if np.abs(axis[0]):
        T[:3, :3] = sy.Matrix([[1, 0, 0],
                               [0, c1, -s1],
                               [0, s1, c1]])
        return T
    if np.abs(axis[1]):
        T[:3, :3] = sy.Matrix([[c1, 0, s1],
                               [0, 1, 0],
                               [-s1, 0, c1]])
        return T
    if np.abs(axis[2]):
        T[:3, :3] = sy.Matrix([[c1, -s1, 0],
                               [s1, c1, 0],
                               [0, 0, 1]])
        return T


def subs_value(vars, vars_value):
    subs_dic = {}
    for i in range(len(vars)):
        for j in range(len(vars[i])):
            subs_dic.update({vars[i][j]: vars_value[i][j]})
    return subs_dic
