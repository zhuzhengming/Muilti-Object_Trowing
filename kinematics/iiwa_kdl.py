import numpy as np
import PyKDL as kdl
from urdf_parser_py.urdf import URDF


import kinematics.kdl_parser as kdl_parser


class iiwa:
    def __init__(self, start_link=None, tip_link=None, right_hand=True, use_fingers=None, path_prefix='',