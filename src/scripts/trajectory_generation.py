import sys
sys.path.append("../")
import time
import math
import rospy
import copy
import numpy as np
import pybullet as p
import pybullet_data
import pdb

from std_msgs.msg import String
from sensor_msgs.msg import JointState
from std_msgs.msg import Int64

from pathlib import Path
from sys import path
from ruckig import InputParameter, Ruckig, Trajectory
from utils.mujoco_interface import Robot

from std_srvs.srv import Trigger
import mujoco
from mujoco import viewer

# global variables
SIMULATION = False  # Set to True to run the simulation before commanding the real robot
REAL_ROBOT_STATE = True  # Set to True to use the real robot state to start the simulation

## ---- ROS conversion and callbacks functions ---- ##
class Throwing_controller:
    def __init__(self, simulator='mujoco'):
        rospy.init_node("throwing_controller", anonymous=True)

        # initialize ROS subscriber/publisher
        self.fsm_state_pub = rospy.Publisher('fsm_state', String, queue_size=1)
        self.target_state_pub = rospy.Publisher('/computed_torque_controller/target_state', JointState,
                                                queue_size=10)  # for debug
        self.command_pub = rospy.Publisher('/iiwa_impedance_joint', JointState, queue_size=10)

        rospy.Subscriber('/iiwa/joint_states', JointState, self.joint_states_callback, queue_size=10)
        rospy.Subscriber('/throw_node/throw_state', Int64, self.scheduler_callback)
        # rospy.wait_for_service('/franka_control/gripper_deactivate')

        # Initialize robot state, to be updated from robot
        self.robot_state = JointState()
        self.robot_state.position = [0.0 for _ in range(7)]
        self.robot_state.velocity = [0.0 for _ in range(7)]
        self.robot_state.effort = [0.0 for _ in range(7)]

        # Initialize target state, to be updated from planner
        self.target_state = JointState()

        self.simulator = simulator

        # Ruckig margins for throwing
        self.MARGIN_VELOCITY = rospy.get_param('/MARGIN_VELOCITY')
        self.MARGIN_ACCELERATION = rospy.get_param('/MARGIN_ACCELERATION')
        self.MARGIN_JERK = rospy.get_param('/MARGIN_JERK')

        # constraints of iiwa 7
        self.max_velocity = np.array(rospy.get_param('/max_velocity'))
        self.max_acceleration = np.array(rospy.get_param('/max_acceleration'))
        self.max_jerk = np.array(rospy.get_param('/max_jerk'))

        # allegro pre-defined pose
        self.allegro_home = np.array(rospy.get_param('/hand_home_pose'))
        self.envelop = np.array(rospy.get_param('/envelop_pose'))

        # q0 is the home state
        if SIMULATION:
            self.q0 = np.array([-0.32032486, 0.02707055, -0.22881525, -1.42611918, 1.38608943, 0.5596685, -1.34659665 + np.pi])
        else:
            robot_state = rospy.wait_for_message("/iiwa/joint_states", JointState)
            self.q0 = np.array(robot_state.position)

        # qs for the initial state
        self.qs = np.array([-0.3217 + 0.5, 0.6498, 0.1635, -1.4926, -0.0098, 0.8557, 1.2881])
        self.qs_dot = np.zeros(7)
        self.qs_dotdot = np.zeros(7)

        # qd for the throwing state
        self.qd = self.qs + np.array([0.0, 0.0, -0.4, 0.0, 0.0, 0.0, 0.0])
        self.qd_dot =       np.array([0.0, 0.0, -0.4, 0.0, 0.0, 0.0, 0.0])
        self.qd_dotdot = np.zeros(7)

        # compute the nominal throwing and slowing trajectory
        self.trajectory = self.get_traj_from_ruckig(self.qs, self.qs_dot, self.qs_dotdot,
                                                    self.qd, self.qd_dot, self.qd_dotdot,
                                                    margin_velocity=self.MARGIN_VELOCITY,
                                                    margin_acceleration=self.MARGIN_ACCELERATION)

        self.trajectory_back = self.get_traj_from_ruckig(self.qd, self.qd_dot, self.qd_dotdot,
                                                         self.qs, self.qs_dot, self.qs_dotdot,
                                                         margin_velocity=self.MARGIN_VELOCITY,
                                                         margin_acceleration=self.MARGIN_ACCELERATION * 0.5)

        if self.trajectory is None or self.trajectory_back is None:
            rospy.logerr("Trajectory is None")

        self.traj_time = self.trajectory.duration
        self.traj_back_time = self.trajectory_back.duration

        if simulator == 'mujoco' and SIMULATION == True:
            xml_path = '../description/iiwa7_allegro_ycb.xml'
            obj_name = ''
            model = mujoco.MjModel.from_xml_path(xml_path)
            data = mujoco.MjData(model)
            mujoco.mj_step(model, data)

            self.view = viewer.launch_passive(model, data)

            obj_names = ['banana', 'bottle', 'chip_can', 'soft_scrub', 'sugar_box']
            num = 0
            obj = obj_names[num]
            self.r = Robot(model, data, self.view, auto_sync=True, obj_names=obj_names)
            self.r.iiwa_hand_go(q=self.q0,qh=self.envelop)

        # Run simulation once for visualization
        if SIMULATION:
            if self.simulator == 'pybullet':
                self.run_simulation_pybullet()
            # elif self.simulator == 'mujoco':
                # self.run_simulation_mujoco()

        self.stamp = []
        self.real_pos = []
        self.real_vel = []
        self.real_eff = []
        self.target_pos = []
        self.target_vel = []
        self.target_eff = []

        while True:
            if SIMULATION:
                self.robot_state.position = np.array(self.r.q)
                self.robot_state.velocity = np.array(self.r.dq)

            print("Got robot state")
            break

        # get parameters and initialization
        self.ERROR_THRESHOLD = rospy.get_param('/ERROR_THRESHOLD')  # Threshold to switch from homing to throwing state
        self.GRIPPER_DELAY = rospy.get_param('/GRIPPER_DELAY')

        self.time_throw = np.inf  # Planned time of throwing
        self.fsm_state = "IDLE"

        self.nIter_time = 0.0
        time.sleep(1.0)


    def run_simulation_pybullet(self):
        clid = p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(cameraDistance=2.5, cameraYaw=145, cameraPitch=-45,
                                     cameraTargetPosition=[0.8, 0, 0])
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        delta_t = 0.002
        p.setTimeStep(delta_t)
        robotId = p.loadURDF("../description/iiwa_description/urdf/iiwa7_lasa.urdf",
                             [0.0, 0.0, 0.0], useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE)
        controlled_joints = [0, 1, 2, 3, 4, 5, 6]
        robotEndEffectorIndex = 6

        if REAL_ROBOT_STATE:
            # get initial robot state from ROS message
            robot_state = rospy.wait_for_message("/iiwa/joint_states", JointState)
            print("initial robot state")

            q0 = np.array(robot_state.position)
            q0_dot = np.array(robot_state.velocity)
            self.qs[0] = q0[0]
            # compute the nominal throwing and slowing trajectory
            self.trajectory = self.get_traj_from_ruckig(self.qs, self.qs_dot, self.qs_dotdot,
                                                   self.qd, self.qd_dot, self.qd_dotdot,
                                                   margin_velocity=self.MARGIN_VELOCITY,
                                                   margin_acceleration=self.MARGIN_ACCELERATION)
            self.trajectory_back = self.get_traj_from_ruckig(self.qd, self.qd_dot, self.qd_dotdot,
                                                        self.qs, self.qs_dot, self.qs_dotdot,
                                                   margin_velocity=self.MARGIN_VELOCITY,
                                                   margin_acceleration=self.MARGIN_ACCELERATION * 0.5)

            if self.trajectory is None or self.trajectory_back is None:
                rospy.logerr("Trajectory is None")

            self.traj_time = self.trajectory.duration
            self.traj_back_time = self.trajectory_back.duration
        else:
            # sample a random starting configuration
            np.random.seed(0)
            q0 = np.random.uniform(0.0, 1.5, 7)
        # reset the robot to the random configuration
        p.resetJointStatesMultiDof(robotId, controlled_joints, [[q0_i] for q0_i in q0],
                                   targetVelocities=[[q0_i] for q0_i in np.zeros(7)])
        # pdb.set_trace(header="PDB PAUSE: Press C to start simulation...")

        # generate the trajectory to go to qs
        trajectory_to_qs = self.get_traj_from_ruckig(q0, np.zeros(7), np.zeros(7),
                                                     self.qs, self.qs_dot, self.qs_dotdot,
                                                     margin_velocity=0.2, margin_acceleration=0.1)

        if trajectory_to_qs is None:
            rospy.logerr("Trajectory is None")

        flag = True
        landing_pos = None
        video_path = None

        # execute homing trajectory
        tt = 0
        while (True):
            ref = trajectory_to_qs.at_time(tt)
            p.resetJointStatesMultiDof(robotId, controlled_joints, [[q0_i] for q0_i in ref[0]],
                                       targetVelocities=[[q0_i] for q0_i in ref[1]])
            # REAL: publish joint velocity command
            p.stepSimulation()
            # pdb.set_trace()
            time.sleep(delta_t)
            tt += delta_t
            if tt > trajectory_to_qs.duration:
                break
        waypoints = []
        tt = 0
        # video_path = "bsa_throw.mp4"
        if not (video_path is None):
            logId = p.startStateLogging(loggingType=p.STATE_LOGGING_VIDEO_MP4, fileName=video_path)
        while (True):
            if flag:
                ref_full = self.trajectory.at_time(tt)

                ref = [ref_full[i][:7] for i in range(3)]
                # ref_base = [ref_full[i][-2:] for i in range(3)]
                p.resetJointStatesMultiDof(robotId, controlled_joints, [[q0_i] for q0_i in ref[0]],
                                           targetVelocities=[[q0_i] for q0_i in ref[1]])
                eef_state = p.getLinkState(robotId, robotEndEffectorIndex, computeLinkVelocity=1)
            else:
                # slow down the robot
                ref_full = self.trajectory_back.at_time(tt - self.traj_time)
                ref = [ref_full[i][:7] for i in range(3)]
                p.resetJointStatesMultiDof(robotId, controlled_joints, [[q0_i] for q0_i in ref[0]],
                                           targetVelocities=[[q0_i] for q0_i in ref[1]])

            p.stepSimulation()
            tt = tt + delta_t
            if tt > self.trajectory.duration and tt <= self.trajectory.duration + self.trajectory_back.duration:
                flag = False

            time.sleep(delta_t)
            if tt > 6.0:
                break
        p.disconnect()

    def run_simulation_mujoco(self):

        if REAL_ROBOT_STATE:
            # get initial robot state from ROS message
            robot_state = rospy.wait_for_message("/iiwa/joint_states", JointState)
            print("initial robot state")

            q0 = np.array(robot_state.position)
            q0_dot = np.array(robot_state.velocity)
            self.qs[0] = q0[0]
            self.qs_dot[0] = q0_dot[0]
            # compute the nominal throwing and slowing trajectory
            self.trajectory = self.get_traj_from_ruckig(self.qs, self.qs_dot, self.qs_dotdot,
                                                   self.qd, self.qd_dot, self.qd_dotdot,
                                                   margin_velocity=self.MARGIN_VELOCITY,
                                                   margin_acceleration=self.MARGIN_ACCELERATION)
            self.trajectory_back = self.get_traj_from_ruckig(self.qd, self.qd_dot, self.qd_dotdot,
                                                        self.qs, self.qs_dot, self.qs_dotdot,
                                                   margin_velocity=self.MARGIN_VELOCITY,
                                                   margin_acceleration=self.MARGIN_ACCELERATION * 0.5)

            if self.trajectory is None or self.trajectory_back is None:
                rospy.logerr("Trajectory is None")

            self.traj_time = self.trajectory.duration
            self.traj_back_time = self.traj_time.duration
        else:
            # get initial robot state from Mujoco
            q0 = np.copy(self.r.q)
            q0_dot = np.copy(self.r.dq)

        qs = np.copy(self.r.q)
        qs[0] += 0.2
        qs_dot = np.copy(self.r.dq)

        # reset the robot to initial configuration
        self.r.iiwa_hand_go(q=q0, d_pose=q0_dot, qh=np.zeros(16))
        # pdb.set_trace(header="PDB PAUSE: Press C to start simulation...")

        # generate the trajectory to go to qs
        trajectory_to_qs = self.get_traj_from_ruckig(q0, q0_dot, np.zeros(7),
                                                     qs, qs_dot, self.qs_dotdot,
                                                     margin_velocity=0.2, margin_acceleration=0.1)

        if trajectory_to_qs is None:
            rospy.logerr("Trajectory is None")

        flag = True
        delta_t = 0.002

        # execute homing trajectory
        tt =0
        while True:
            ref = trajectory_to_qs.at_time(tt)
            self.r.iiwa_hand_go(q=ref[0], d_pose=ref[1], qh=np.zeros(16))

            time.sleep(delta_t)
            tt += delta_t
            if tt > trajectory_to_qs.duration:
                break

        waypoints = []
        tt = 0
        while True:
            if flag:
                ref_full = self.trajectory.at_time(tt)
                ref = [ref_full[i][:7] for i in range(3)]

                self.r.iiwa_hand_go(q=ref[0], d_pose=ref[1], qh=np.zeros(16))
            else:
                # slow down the robot
                ref_full = self.trajectory_back.at_time(tt - self.traj_time)
                ref = [ref_full[i][:7] for i in range(3)]

                self.r.iiwa_hand_go(q=ref[0], d_pose=ref[1], qh=np.zeros(16))

            tt += delta_t
            if tt > self.trajectory.duration and tt <= self.trajectory.duration + self.trajectory_back.duration:
                flag = False
            time.sleep(delta_t)
            if tt > 6.0:
                break
        # self.view.close()

    def save_tracking_data_to_npy(self):

        filename = '../output/data/throwing.npy'
        # Save data to npy files
        np.save(filename, {'stamp': self.stamp,
                            'real_pos': self.real_pos,
                           'real_vel': self.real_vel,
                           'real_eff': self.real_eff,
                           'target_pos': self.target_pos,
                           'target_vel': self.target_vel,
                           'target_eff': self.target_eff})
        print("Tracking data saved to npy files.")

    def run(self, start_time):
        # ------------ Control Loop ------------ #
        dT = 1e-3
        rate = rospy.Rate(1.0 / dT)
        cycle = 0

        while not rospy.is_shutdown():
            # Publish state and fsm for debug
            self.fsm_state_pub.publish(self.fsm_state)

            # update robot state
            if SIMULATION:
                q_cur = self.r.q
                q_cur_dot = self.r.dq
                q_cur_effort = np.zeros(7)
            else:
                # self.robot_state = rospy.wait_for_message("/iiwa/joint_states", JointState)
                q_cur = np.array(self.robot_state.position)
                q_cur_dot = np.array(self.robot_state.velocity)
                q_cur_effort = np.array(self.robot_state.effort)


            if self.fsm_state == "IDLE":
                # pdb.set_trace(header="Press C to start homing...")
                # print("HOMING...")

                self.homing_traj = self.get_traj_from_ruckig(q_cur, q_cur_dot, np.zeros(7),
                                                             self.qs, self.qs_dot, self.qs_dotdot,
                                                             margin_velocity=self.MARGIN_VELOCITY,
                                                             margin_acceleration=self.MARGIN_ACCELERATION)
                if self.homing_traj is None:
                    rospy.logerr("Trajectory is None")

                # print("IDLING: initial state", self.q0, "homing trajectory duration", self.homing_traj.duration)

                # update state
                self.fsm_state = "HOMING"
                self.time_start_homing = rospy.get_time()


            elif self.fsm_state == "HOMING":

                # Activate integrator term when close to target
                error_position = np.array(q_cur) - np.array(self.qs)
                if np.linalg.norm(error_position) < self.ERROR_THRESHOLD:
                    # Jump to next state
                    self.fsm_state = "IDLE_THROWING"
                    time.sleep(0.5)
                    # print("IDLE_THROWING")
                    # pdb.set_trace(header="Press C to see the throwing trajectory...")
                    self.scheduler_callback(Int64(1))

                time_now = rospy.get_time()
                ref = self.homing_traj.at_time(time_now - self.time_start_homing)
                self.target_state.header.stamp = time_now
                self.target_state.position = ref[0]
                self.target_state.velocity = ref[1]
                self.target_state.effort = ref[2]

                if SIMULATION:
                    self.r.iiwa_hand_go(q=self.target_state.position,
                                        d_pose=self.target_state.velocity,
                                        qh=np.zeros(16))
                else:
                    self.target_state_pub.publish(self.target_state)
                    self.command_pub.publish(self.convert_command_to_ROS(time_now, ref[0], ref[1], ref[2]))

            elif self.fsm_state == "IDLE_THROWING":
                continue

            elif self.fsm_state == "THROWING":
                # execute throwing trajectory
                time_now = rospy.get_time()
                # release gripper
                # if time_now - self.time_start_throwing > self.throwing_traj.duration - self.GRIPPER_DELAY:
                #     threading.Thread(target=self.deactivate_gripper).start()

                if time_now - self.time_start_throwing > self.throwing_traj.duration - dT:
                    self.fsm_state = "SLOWING"

                    self.trajectory_back = self.get_traj_from_ruckig(q_cur, q_cur_dot, self.qd_dotdot,
                                                                     self.qs, self.qs_dot, self.qs_dotdot,
                                                                margin_velocity=self.MARGIN_VELOCITY * 0.2,
                                                                margin_acceleration=self.MARGIN_ACCELERATION * 0.2)
                    if self.trajectory_back is None:
                        rospy.logerr("Trajectory is None")

                    # print("SLOWING")
                    # pdb.set_trace(header="Press C to see the slowing trajectory...")
                    self.time_start_slowing = time_now

                ref = self.throwing_traj.at_time(time_now - self.time_start_throwing)
                self.target_state.header.stamp = time_now
                self.target_state.position = ref[0]
                self.target_state.velocity = ref[1]
                self.target_state.effort = ref[2]

                if SIMULATION:
                    self.r.iiwa_hand_go(q=self.target_state.position,
                                        d_pose=self.target_state.velocity,
                                        qh=np.zeros(16))
                else:
                    self.target_state_pub.publish(self.target_state)
                    self.command_pub.publish(self.convert_command_to_ROS(time_now, ref[0], ref[1], ref[2]))

                # record error
                if(self.time_throw - (time_now - self.time_start_throwing) < 0.05 * self.time_throw):

                    self.stamp.append(time_now)

                    self.real_pos.append(q_cur)
                    self.real_vel.append(q_cur_dot)
                    self.real_eff.append(q_cur_effort)
                    self.target_pos.append(self.target_state.position)
                    self.target_vel.append(self.target_state.velocity)
                    self.target_eff.append(self.target_state.effort)

            elif self.fsm_state == "SLOWING":
                time_now = rospy.get_time()

                if time_now - self.time_start_slowing > self.trajectory_back.duration - dT:
                    self.save_tracking_data_to_npy()
                    cycle += 1
                    break

                ref = self.trajectory_back.at_time(time_now - self.time_start_slowing)
                self.target_state.header.stamp = time_now
                self.target_state.position = ref[0]
                self.target_state.velocity = ref[1]
                self.target_state.effort = ref[2]

                if SIMULATION:
                    self.r.iiwa_hand_go(q=self.target_state.position,
                                        d_pose=self.target_state.velocity,
                                        qh=np.zeros(16))
                else:
                    self.target_state_pub.publish(self.target_state)
                    self.command_pub.publish(self.convert_command_to_ROS(time_now, ref[0], ref[1], ref[2]))

            rate.sleep()

    ## ---- ROS conversion and callbacks functions ---- ##
    def convert_command_to_ROS(self, time_now, qd, qd_dot, qd_dotdot):
        command = JointState()
        command.header.stamp = rospy.Time.from_sec(time_now)
        command.name = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6',
                        'panda_joint7']
        command.position =  qd
        command.velocity = qd_dot
        command.effort = qd_dotdot

        return command

    def joint_states_callback(self, state):
        self.robot_state.header = copy.deepcopy(state.header)
        self.robot_state.position = copy.deepcopy(state.position)
        self.robot_state.velocity = copy.deepcopy(state.velocity)
        self.robot_state.effort = copy.deepcopy(state.effort)

    def scheduler_callback(self, msg):
        # print("scheduler msg", msg)
        if self.fsm_state == "IDLE_THROWING" and msg.data == 1:
            # compute new trajectory to throw from current position
            if SIMULATION:
                q_cur = self.r.q
                q_cur_dot = self.r.dq
            else:
                q_cur = np.array(self.robot_state.position)
                q_cur_dot = np.array(self.robot_state.velocity)

            self.throwing_traj = self.get_traj_from_ruckig(q_cur, q_cur_dot, np.zeros(7),
                                                           self.qd, self.qd_dot, self.qd_dotdot,
                                                      margin_velocity=self.MARGIN_VELOCITY,
                                                      margin_acceleration=self.MARGIN_ACCELERATION)
            if self.throwing_traj is None:
                rospy.logerr("Trajectory is None")

            self.time_start_throwing = rospy.get_time()
            self.time_throw = self.throwing_traj.duration
            self.fsm_state = "THROWING"
            # print("Throwing...")

    def deactivate_gripper(self):
        rospy.wait_for_service('/franka_control/gripper_deactivate')
        try:
            gripper_deactivate = rospy.ServiceProxy('/franka_control/gripper_deactivate', Trigger)
            gripper_deactivate()
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)


    # Path to the build directory including a file similar to 'ruckig.cpython-37m-x86_64-linux-gnu'.
    build_path = Path(__file__).parent.absolute().parent / 'build'
    path.insert(0, str(build_path))

    def get_traj_from_ruckig(self, q0, q0_dot, q0_dotdot,
                             qd, qd_dot, qd_dotdot,
                             margin_velocity=1.0, margin_acceleration=0.7,
                             margin_jerk=None):

        """
            Generates a smooth trajectory using the Ruckig algorithm for the given joint states.

            Parameters:
            - q0: Initial joint positions
            - q0_dot: Initial joint velocities
            - q0_dotdot: Initial joint accelerations
            - qd: Target joint positions
            - qd_dot: Target joint velocities
            - qd_dotdot: Target joint accelerations
            - margin_velocity: Velocity margin
            - margin_acceleration: Acceleration margin
            - margin_jerk: Jerk margin

            Returns:
            - trajectory: The generated trajectory object
            """

        if margin_jerk is None:
            margin_jerk = self.MARGIN_JERK

        inp = InputParameter(len(q0))
        inp.current_position = q0
        inp.current_velocity = q0_dot
        inp.current_acceleration = q0_dotdot

        inp.target_position = qd
        inp.target_velocity = qd_dot
        inp.target_acceleration = qd_dotdot

        inp.max_velocity = self.max_velocity * margin_velocity
        inp.max_acceleration = self.max_acceleration * margin_acceleration
        inp.max_jerk = self.max_jerk * margin_jerk

        otg = Ruckig(len(q0))
        trajectory = Trajectory(len(q0))
        _ = otg.calculate(inp, trajectory)

        return trajectory


if __name__ == '__main__':

    simulation_mode = 'mujoco'
    throwing_controller = Throwing_controller(simulator=simulation_mode)
    start_time = rospy.get_time()

    for nTry in range(100):
        # print("test number", nTry + 1)

        throwing_controller.fsm_state = "IDLE"
        throwing_controller.run(start_time)

        time.sleep(1)

        # Stop controller when ROS is stopped
        if rospy.is_shutdown():
            break