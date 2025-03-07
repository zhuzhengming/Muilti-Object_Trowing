import sys
sys.path.append("../")
import time
import numpy as np
from utils.mujoco_interface import Robot
import mujoco
from mujoco import viewer
import tools.rotations as rot

xml_path = '../description/iiwa7_allegro_ycb.xml'
obj_name = ''
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
mujoco.mj_step(model, data)

view = viewer.launch_passive(model, data)

obj_names = ['banana', 'bottle', 'chip_can', 'soft_scrub', 'suger_box']
r = Robot(model, data, view, auto_sync=True, obj_names=obj_names)

# print('obj pose', r.x_obj)

tmp = np.array([0, 0, 0, 0])
q_middle = tmp + np.array([0, 0, 0, 0])
q_ring = tmp + np.array([0, 0, 0, 0])
q_index = tmp
q_thumb = tmp
qh = np.concatenate([q_index, q_middle, q_ring, q_thumb])
q_all = np.concatenate([np.array([0, 0, 0, 0, 0, 0, np.pi]), qh])
q0 = np.copy(r.q)
qd = np.copy(q0)

t0 = time.time()
while 1:
    t = time.time() - t0
    qd[0] = q0[0] + 1 * np.sin(2 *np.pi * 1 * t)

    r.iiwa_hand_go(qd, qh)
    time.sleep(0.005)
