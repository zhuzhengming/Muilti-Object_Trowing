import sys
sys.path.append('../')
import copy
import numpy as np
from utils.controller_utils import Robot
import matplotlib

home_pose = np.array([ 0.03063398, -0.15441399,  0.82104033,  0.82512679,  0.00664029,
       -0.2039733 ,  0.80409743,  0.        , -0.10271061, -0.06070025,
        0.90650362,  0.81377007,  0.86300686,  0.44554398,  0.08874171,
        0.77574977])

catch_pose = np.array([-0.50889807,  1.74874918,  0.12737114,  0.39226041,  0.07008444,
        1.64808905,  1.36754356,  0.21940653,  0.0937576 ,  1.77053021,
        1.36580765,  0.55991777,  0.94150295,  0.14131779, -0.24060529,
        0.96755097])

envelop_pose = np.array([ 0.07941064,  1.77857954,  0.44877399,  0.62444235,  0.29191241,
        1.55038098,  0.61655537,  0.36046541,  0.4311105 ,  1.76264807,
        1.02352552,  0.9226552 ,  1.54144452,  0.48900141, -0.08175917,
        1.3566044 ])



def singleObject_throwing():
    target_pose = r.x
    # target_pose[0] += 0.15
    target_pose[2] += 0.25

    r.iiwa_cartesion_impedance_control(target_pose, vel=1.5)

    threshold = 0.1
    reach = False
    while not reach:
        if np.linalg.norm(r.x[:3] - target_pose[:3]) < threshold:
            r.move_to_joints(home_pose, vel=[0.2, 8.0])
            reach = True
        else:
            continue

def go_home(iiwa_home_pose):
    r.iiwa_cartesion_impedance_control(iiwa_home_pose, vel=1.0)
    r.move_to_joints(envelop_pose, vel=[0.2, 8.0])


if __name__ == '__main__':
    r = Robot(optitrack_frame_names=['iiwa_base7', 'realsense_m'], position_control=False)

    iiwa_home_pose = copy.deepcopy(r.x)

    while True:
        print("\n Choose an action:")
        print("1: Go home")
        print("2: Throwing")

        user_input = input("Enter your choice: ")

        if user_input == '1':
            go_home(iiwa_home_pose)
        elif user_input == '2':
            singleObject_throwing()
        else:
            print("Invalid input")
            continue



