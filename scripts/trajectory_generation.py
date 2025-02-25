import argparse
import time
import math
import pickle
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
from ruckig import InputParameter, Ruckig, Trajectory, Result

from std_srvs.srv import Trigger
import threading


def deactivate_gripper():
    rospy.wait_for_service('/franka_control/gripper_deactivate')
    try:
        gripper_deactivate = rospy.ServiceProxy('/franka_control/gripper_deactivate', Trigger)
        gripper_deactivate()
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s", e)


SIMULATION = True  # Set to True to run the simulation before commanding the real robot
REAL_ROBOT_STATE = True  # Set to True to use the real robot state to start the simulation
ERROR_THRESHOLD = 0.5  # Threshold to switch from homing to throwing state
GRIPPER_DELAY = 350e-3

# Ruckig margins for throwing
MARGIN_VELOCITY = 1.2
MARGIN_ACCELERATION = 0.6
MARGIN_JERK = 0.01
# Path to the build directory including a file similar to 'ruckig.cpython-37m-x86_64-linux-gnu'.
build_path = Path(__file__).parent.absolute().parent / 'build'
path.insert(0, str(build_path))


def get_traj_from_ruckig(q0, q0_dot, q0_dotdot, qd, qd_dot, qd_dotdot, margin_velocity=1.0, margin_acceleration=0.7,
                         margin_jerk=MARGIN_JERK):
    inp = InputParameter(3)
    inp.current_position = q0
    inp.current_velocity = q0_dot
    inp.current_acceleration = q0_dotdot

    inp.target_position = qd
    inp.target_velocity = qd_dot
    inp.target_acceleration = qd_dotdot

    inp.max_velocity = np.array([5.0, 5.0, 5.0]) * margin_velocity
    inp.max_acceleration = np.array([15, 7.5, 10]) * margin_acceleration
    inp.max_jerk = np.array([7500, 3750, 5000]) * margin_jerk

    otg = Ruckig(3)
    trajectory = Trajectory(3)
    _ = otg.calculate(inp, trajectory)
    return trajectory


# set the boundary states
# qs for the initial state and qd for the throwing state
qs = np.zeros(3)
qs[1] = -0.5
# qs[1] = -math.pi# -0.5*math.pi
qs_dot = np.zeros(3)
qs_dotdot = np.zeros(3)
qd = np.array([0.0, 1.5 - 2.0 * math.pi, 1.0])
# qd = np.array([0.0,1.5,1.0])

qd_dot = np.array([0.0, -4.0, -4.0]) * MARGIN_VELOCITY
qd_dotdot = np.array([0.0, 0.0, 0.0])

# compute the nominal throwing and slowing trajectory
trajectory = get_traj_from_ruckig(qs, qs_dot, qs_dotdot, qd, qd_dot, qd_dotdot, margin_velocity=MARGIN_VELOCITY,
                                  margin_acceleration=MARGIN_ACCELERATION)
trajectory_back = get_traj_from_ruckig(qd, qd_dot, qd_dotdot, qs, qs_dot, qs_dotdot, margin_velocity=MARGIN_VELOCITY,
                                       margin_acceleration=MARGIN_ACCELERATION * 0.5)

traj_time = trajectory.duration
traj_back_time = trajectory_back.duration


## ---- ROS conversion and callbacks functions ---- ##
class Throwing_controller:
    def __init__(self):
        # Init ROS and subscribers/publishers
        rospy.init_node("throwing_controller_elastic_arm", anonymous=True)
        # Run simulation once for visualization
        if SIMULATION:
            self.run_simulation()
        while True:
            self.robot_state = rospy.wait_for_message("/franka_state_controller/joint_states", JointState)
            print("Got robot state", self.robot_state.position[-3:])
            break
        self.fsm_state_pub = rospy.Publisher('fsm_state', String, queue_size=1)
        # self.command_pub = rospy.Publisher('/robot/arm/computed_torque_controller/command', JointState, queue_size=1)
        self.target_state_pub = rospy.Publisher('/robot/arm/computed_torque_controller/target_state', JointState,
                                                queue_size=1)
        self.command_pub = rospy.Publisher('/joint_velocity_controller_darko/command', JointState, queue_size=1)
        # rospy.Subscriber('/robot/arm/computed_torque_controller/robot_state', JointState, self.joint_states_callback, queue_size=1)
        rospy.Subscriber('/franka_state_controller/joint_states', JointState, self.joint_states_callback, queue_size=1)

        rospy.Subscriber('/throw_node/throw_state', Int64, self.scheduler_callback)
        rospy.wait_for_service('/franka_control/gripper_deactivate')

        self.time_throw = np.inf  # Planned time of throwing
        self.fsm_state = "IDLE"

        self.nIter_time = 0.0
        time.sleep(1.0)

        # Initialize robot state, to be updated from robot
        self.robot_state = JointState()
        self.robot_state.position = [0.0 for _ in range(7)]
        self.robot_state.velocity = [0.0 for _ in range(7)]
        self.robot_state.effort = [0.0 for _ in range(7)]

        # Initialize target state, to be updated from planner
        self.target_state = JointState()

        self.stamp = []
        self.tracking_error_pos = []
        self.tracking_error_vel = []
        self.joint_velo_his = []

    def simulate_trajectory(self, trajectory):
        clid = p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(cameraDistance=2.5, cameraYaw=145, cameraPitch=-45,
                                     cameraTargetPosition=[0.8, 0, 0])

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        p.setGravity(0, 0, -9.81)
        delta_t = 0.002
        p.setTimeStep(delta_t)
        robotId = p.loadURDF("descriptions/URDF_Assembly_Arm_v0_1_SLDASM/urdf/TUMArm_v0_1_SLDASM_Robotiq.urdf",
                             [0.0, 0.0, 0.0], useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE)
        controlled_joints = [0, 1, 2]
        robotEndEffectorIndex = 15
        planeId = p.loadURDF("plane.urdf", [0, 0, 0.0])
        soccerballId = p.loadURDF("soccerball.urdf", [-3.0, 0, 3], globalScaling=0.05)
        # move the ball away from the robot at the beginning
        p.resetBasePositionAndOrientation(soccerballId, [100, 100, 100], [0, 0, 0, 1])

        # reset the robot to the initial configuration of the trajectory
        q0 = trajectory.at_time(0)[0]
        p.resetJointStatesMultiDof(robotId, controlled_joints, [[q0_i] for q0_i in q0],
                                   targetVelocities=[[q0_i] for q0_i in np.zeros(3)])

        # play the trajectory twice
        tt = 0
        counter = 0
        while (True):
            ref = trajectory.at_time(tt)
            p.resetJointStatesMultiDof(robotId, controlled_joints, [[q0_i] for q0_i in ref[0]],
                                       targetVelocities=[[q0_i] for q0_i in ref[1]])
            p.stepSimulation()
            time.sleep(delta_t)
            tt += delta_t
            if tt > trajectory.duration:
                tt = 0
                counter += 1
                if counter == 2:
                    break

        p.disconnect()

    def run_simulation(self):
        clid = p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(cameraDistance=2.5, cameraYaw=145, cameraPitch=-45,
                                     cameraTargetPosition=[0.8, 0, 0])

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        p.setGravity(0, 0, -9.81)
        delta_t = 0.002
        p.setTimeStep(delta_t)
        robotId = p.loadURDF("descriptions/URDF_Assembly_Arm_v0_1_SLDASM/urdf/TUMArm_v0_1_SLDASM_Robotiq.urdf",
                             [0.0, 0.0, 0.0], useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE)
        controlled_joints = [0, 1, 2]
        robotEndEffectorIndex = 15
        planeId = p.loadURDF("plane.urdf", [0, 0, 0.0])
        soccerballId = p.loadURDF("soccerball.urdf", [-3.0, 0, 3], globalScaling=0.05)
        # move the ball away from the robot at the beginning
        p.resetBasePositionAndOrientation(soccerballId, [100, 100, 100], [0, 0, 0, 1])

        eef_state = p.getLinkState(robotId, robotEndEffectorIndex, computeLinkVelocity=1)

        if REAL_ROBOT_STATE:
            # get initial robot state from ROS message
            robot_state = rospy.wait_for_message("/franka_state_controller/joint_states", JointState)
            print("initial robot state", robot_state.position[-3:], robot_state.velocity[-3:])
            q0 = np.array(robot_state.position)[-3:]
            q0_dot = np.array(robot_state.velocity)[-3:]
            qs[0] = q0[0]
            # compute the nominal throwing and slowing trajectory
            trajectory = get_traj_from_ruckig(qs, qs_dot, qs_dotdot, qd, qd_dot, qd_dotdot,
                                              margin_velocity=MARGIN_VELOCITY, margin_acceleration=MARGIN_ACCELERATION)
            trajectory_back = get_traj_from_ruckig(qd, qd_dot, qd_dotdot, qs, qs_dot, qs_dotdot,
                                                   margin_velocity=MARGIN_VELOCITY,
                                                   margin_acceleration=MARGIN_ACCELERATION * 0.5)

            traj_time = trajectory.duration
            traj_back_time = trajectory_back.duration
        else:
            # sample a random starting configuration
            np.random.seed(0)
            q0 = np.random.uniform(0.0, 2.0, 3)
        # reset the robot to the random configuration
        p.resetJointStatesMultiDof(robotId, controlled_joints, [[q0_i] for q0_i in q0],
                                   targetVelocities=[[q0_i] for q0_i in np.zeros(3)])
        pdb.set_trace(header="PDB PAUSE: Press C to start simulation...")

        # generate the trajectory to go to qs
        trajectory_to_qs = get_traj_from_ruckig(q0, np.zeros(3), np.zeros(3), qs, qs_dot, qs_dotdot,
                                                margin_velocity=0.2, margin_acceleration=0.1)

        recordind_flag = True
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
                ref_full = trajectory.at_time(tt)

                ref = [ref_full[i][:7] for i in range(3)]
                # ref_base = [ref_full[i][-2:] for i in range(3)]
                p.resetJointStatesMultiDof(robotId, controlled_joints, [[q0_i] for q0_i in ref[0]],
                                           targetVelocities=[[q0_i] for q0_i in ref[1]])
                eef_state = p.getLinkState(robotId, robotEndEffectorIndex, computeLinkVelocity=1)
                p.resetBasePositionAndOrientation(soccerballId, eef_state[0], [0, 0, 0, 1])
                p.resetBaseVelocity(soccerballId, linearVelocity=eef_state[-2])
            else:
                # slow down the robot
                ref_full = trajectory_back.at_time(tt - traj_time)
                ref = [ref_full[i][:7] for i in range(3)]
                p.resetJointStatesMultiDof(robotId, controlled_joints, [[q0_i] for q0_i in ref[0]],
                                           targetVelocities=[[q0_i] for q0_i in ref[1]])
            ref_vector = np.zeros(10)
            ref_vector[0] = tt
            ref_vector[1:4] = ref_full[0]
            ref_vector[4:7] = ref_full[1]
            ref_vector[7:10] = ref_full[2]
            waypoints.append(ref_vector)
            p.stepSimulation()
            tt = tt + delta_t
            if tt > trajectory.duration and tt <= trajectory.duration + trajectory_back.duration:
                flag = False

            time.sleep(delta_t)
            if tt > 6.0:
                # tt = 0.0
                # flag = True
                # eef_state = p.getLinkState(robotId, robotEndEffectorIndex, computeLinkVelocity=1)
                # p.resetBasePositionAndOrientation(soccerballId, eef_state[0], [0, 0, 0, 1])
                # if recordind_flag and not (video_path is None):
                #     p.stopStateLogging(logId)
                #     recordind_flag = False
                break
        ball_state = p.getBasePositionAndOrientation(soccerballId)
        # get landing point
        if ball_state[0][2] < 0.025 and landing_pos is None:
            landing_pos = ball_state[0]
        print("Ball has landed", ball_state[0])
        # if not (video_path is None):
        #     p.stopStateLogging(logId)
        p.disconnect()

    def run(self):
        # ------------ Control Loop ------------ #
        dT = 2e-3
        rate = rospy.Rate(1.0 / dT)
        while not rospy.is_shutdown():
            # Publish state and fsm for debug
            self.fsm_state_pub.publish(self.fsm_state)

            # Check tracking error before throw
            if self.fsm_state == "THROWING" or self.fsm_state == "RELEASE":
                if (self.time_throw - rospy.get_time()) < 10e-3:
                    self.tracking_error_pos.append(
                        np.array(self.target_state.position) - np.array(self.robot_state.position)[-3:])
                    self.tracking_error_vel.append(
                        np.array(self.target_state.velocity) - np.array(self.robot_state.velocity)[-3:])
                    self.joint_velo_his.append(np.array(self.robot_state.velocity)[-3:])

            ## ---------------- FSM ---------------- ##
            # # Setup or reset variables prior to throwing ##
            # if self.fsm_state == "WAIT_SIGNAL":
            #     continue

            if self.fsm_state == "IDLE":
                pdb.set_trace(header="PDB PAUSE: Press C to start homing...")
                print("HOMING...")
                current_robot_state = rospy.wait_for_message("/franka_state_controller/joint_states", JointState)
                self.q0 = np.array(current_robot_state.position)[-3:]
                self.homing_traj = get_traj_from_ruckig(self.q0, np.zeros(3), np.zeros(3), qs, qs_dot, qs_dotdot,
                                                        margin_velocity=0.2, margin_acceleration=0.1)
                print("IDLING: initial state", self.q0, "homing trajectory duration", self.homing_traj.duration)
                self.fsm_state = "HOMING"
                self.time_start_homing = rospy.get_time()

            elif self.fsm_state == "HOMING":

                # Activate integrator term when close to target
                error_position = np.array(self.robot_state.position)[-3:] - np.array(qs)
                if np.linalg.norm(error_position) < ERROR_THRESHOLD:
                    # Jump to next state
                    self.fsm_state = "IDLE_THROWING"
                    print("IDLE_THROWING")
                    # pdb.set_trace(header="Press C to see the throwing trajectory...")
                    self.scheduler_callback(Int64(1))

                time_now = rospy.get_time()
                ref = self.homing_traj.at_time(time_now - self.time_start_homing)
                self.target_state.header.stamp = time_now
                self.target_state.position = ref[0]
                self.target_state.velocity = ref[1]
                self.target_state.effort = ref[2]
                self.target_state_pub.publish(self.target_state)
                self.command_pub.publish(self.convert_command_to_ROS(time_now, ref[0], ref[1], ref[2]))

            elif self.fsm_state == "IDLE_THROWING":
                continue

            elif self.fsm_state == "THROWING":
                # execute throwing trajectory
                time_now = rospy.get_time()
                if time_now - self.time_start_throwing > self.throwing_traj.duration - GRIPPER_DELAY:
                    threading.Thread(target=deactivate_gripper).start()

                if time_now - self.time_start_throwing > self.throwing_traj.duration - dT:
                    self.fsm_state = "SLOWING"
                    self.time_start_slowing = time_now
                    q_cur = np.array(self.robot_state.position)[-3:]
                    qdot_cur = np.array(self.robot_state.velocity)[-3:]
                    self.trajectory_back = get_traj_from_ruckig(q_cur, qdot_cur, qd_dotdot, qs, qs_dot, qs_dotdot,
                                                                margin_velocity=MARGIN_VELOCITY,
                                                                margin_acceleration=MARGIN_ACCELERATION * 0.5)
                    print("Slowing...")

                ref = self.throwing_traj.at_time(time_now - self.time_start_throwing)
                self.target_state.header.stamp = time_now
                self.target_state.position = ref[0]
                self.target_state.velocity = ref[1]
                self.target_state.effort = ref[2]
                self.target_state_pub.publish(self.target_state)
                self.command_pub.publish(self.convert_command_to_ROS(time_now, ref[0], ref[1], ref[2]))

            elif self.fsm_state == "SLOWING":
                time_now = rospy.get_time()

                if time_now - self.time_start_slowing > self.trajectory_back.duration - dT:
                    break

                ref = self.trajectory_back.at_time(time_now - self.time_start_slowing)
                self.target_state.header.stamp = time_now
                self.target_state.position = ref[0]
                self.target_state.velocity = ref[1]
                self.target_state.effort = ref[2]
                self.target_state_pub.publish(self.target_state)
                self.command_pub.publish(self.convert_command_to_ROS(time_now, ref[0], ref[1], ref[2]))

            rate.sleep()

    ## ---- ROS conversion and callbacks functions ---- ##
    def convert_command_to_ROS(self, time_now, qd, qd_dot, qd_dotdot):
        command = JointState()
        command.header.stamp = rospy.Time.from_sec(time_now)
        command.name = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6',
                        'panda_joint7']
        command.position = [0.0 for _ in range(4)] + qd
        command.velocity = [0.0 for _ in range(4)] + qd_dot
        command.effort = [0.0 for _ in range(4)] + qd_dotdot

        return command

    def joint_states_callback(self, state):
        self.robot_state.header = copy.deepcopy(state.header)
        self.robot_state.position = copy.deepcopy(state.position)
        self.robot_state.velocity = copy.deepcopy(state.velocity)
        self.robot_state.effort = copy.deepcopy(state.effort)

    def scheduler_callback(self, msg):
        print("scheduler msg", msg)
        if self.fsm_state == "IDLE_THROWING" and msg.data == 1:
            # computue new trajectory to throw from current position
            q_cur = np.array(self.robot_state.position)[-3:]
            qdot_cur = np.array(self.robot_state.velocity)[-3:]
            self.throwing_traj = get_traj_from_ruckig(q_cur, qdot_cur, np.zeros(3), qd, qd_dot, qd_dotdot,
                                                      margin_velocity=MARGIN_VELOCITY,
                                                      margin_acceleration=MARGIN_ACCELERATION)
            # inspect the throwing trajectory
            # self.simulate_trajectory(self.throwing_traj)
            # pdb.set_trace(header="PDB PAUSE: Press C to throw...")
            self.time_start_throwing = rospy.get_time()
            self.fsm_state = "THROWING"
            print("Throwing...")


if __name__ == '__main__':
    throwing_controller = Throwing_controller()

    for nTry in range(100):
        print("test number", nTry + 1)

        throwing_controller.fsm_state = "IDLE"
        throwing_controller.run()

        # Stop controller when ROS is stopped
        if rospy.is_shutdown():
            break