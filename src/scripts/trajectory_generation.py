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

from std_srvs.srv import Trigger
import threading

# global variables
SIMULATION = True  # Set to True to run the simulation before commanding the real robot
REAL_ROBOT_STATE = False  # Set to True to use the real robot state to start the simulation

## ---- ROS conversion and callbacks functions ---- ##
class Throwing_controller:
    def __init__(self):
        rospy.init_node("throwing_controller", anonymous=True)

        # Ruckig margins for throwing
        self.MARGIN_VELOCITY = rospy.get_param('/MARGIN_VELOCITY')
        self.MARGIN_ACCELERATION = rospy.get_param('/MARGIN_ACCELERATION')
        self.MARGIN_JERK = rospy.get_param('/MARGIN_JERK')

        # constraints of iiwa 7
        self.max_velocity = np.array(rospy.get_param('/max_velocity'))
        self.max_acceleration = np.array(rospy.get_param('/max_acceleration'))
        self.max_jerk = np.array(rospy.get_param('/max_jerk'))

        # qs for the initial state and qd for the throwing state
        self.qs = np.zeros(3)
        self.qs[1] = -0.1
        self.qs_dot = np.zeros(3)
        self.qs_dotdot = np.zeros(3)

        self.qd = np.array([0.0, 1.5 - 2.0 * math.pi, 1.0])
        self.qd_dot = np.array([0.0, -1.0, -1.0]) * self.MARGIN_VELOCITY
        self.qd_dotdot = np.array([0.0, 0.0, 0.0])

        # compute the nominal throwing and slowing trajectory
        self.trajectory = self.get_traj_from_ruckig(self.qs, self.qs_dot, self.qs_dotdot, self.qd, self.qd_dot,
                                                    self.qd_dotdot,
                                                    margin_velocity=self.MARGIN_VELOCITY,
                                                    margin_acceleration=self.MARGIN_ACCELERATION)
        self.trajectory_back = self.get_traj_from_ruckig(self.qd, self.qd_dot, self.qd_dotdot, self.qs, self.qs_dot,
                                                         self.qs_dotdot,
                                                         margin_velocity=self.MARGIN_VELOCITY,
                                                         margin_acceleration=self.MARGIN_ACCELERATION * 0.5)

        self.traj_time = self.trajectory.duration
        self.traj_back_time = self.trajectory_back.duration

        # Run simulation once for visualization
        if SIMULATION:
            self.run_simulation()
        while True:
            self.robot_state = rospy.wait_for_message("/iiwa/joint_states", JointState)
            print("Got robot state")
            break

        # initialize ROS subscriber/publisher
        self.fsm_state_pub = rospy.Publisher('fsm_state', String, queue_size=1)
        self.target_state_pub = rospy.Publisher('/computed_torque_controller/target_state', JointState, queue_size=10) # for debug
        self.command_pub = rospy.Publisher('/iiwa_impedance_joint', JointState, queue_size=10)

        rospy.Subscriber('/iiwa/joint_states', JointState, self.joint_states_callback, queue_size=1)
        rospy.Subscriber('/throw_node/throw_state', Int64, self.scheduler_callback)
        # rospy.wait_for_service('/franka_control/gripper_deactivate')

        # get parameters and initialization
        self.ERROR_THRESHOLD = rospy.get_param('/ERROR_THRESHOLD')  # Threshold to switch from homing to throwing state
        self.GRIPPER_DELAY = rospy.get_param('/GRIPPER_DELAY')

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

    # simulation in Pybullet
    def simulate_trajectory(self, trajectory=None):
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
        # reset the robot to the initial configuration of the trajectory
        q0 = self.trajectory.at_time(0)[0]
        p.resetJointStatesMultiDof(robotId, controlled_joints, [[q0_i] for q0_i in q0],
                                   targetVelocities=[[q0_i] for q0_i in np.zeros(7)])

        # play the trajectory twice
        tt = 0
        counter = 0
        while (True):
            ref = self.trajectory.at_time(tt)
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
            trajectory = self.get_traj_from_ruckig(self.qs, self.qs_dot, self.qs_dotdot,
                                                   self.qd, self.qd_dot, self.qd_dotdot,
                                                   margin_velocity=self.MARGIN_VELOCITY,
                                                   margin_acceleration=self.MARGIN_ACCELERATION)
            trajectory_back = self.get_traj_from_ruckig(self.qd, self.qd_dot, self.qd_dotdot,
                                                        self.qs, self.qs_dot, self.qs_dotdot,
                                                   margin_velocity=self.MARGIN_VELOCITY,
                                                   margin_acceleration=self.MARGIN_ACCELERATION * 0.5)

            traj_time = trajectory.duration
            traj_back_time = trajectory_back.duration
        else:
            # sample a random starting configuration
            np.random.seed(0)
            q0 = np.random.uniform(0.0, 1.5, 7)
        # reset the robot to the random configuration
        p.resetJointStatesMultiDof(robotId, controlled_joints, [[q0_i] for q0_i in q0],
                                   targetVelocities=[[q0_i] for q0_i in np.zeros(7)])
        pdb.set_trace(header="PDB PAUSE: Press C to start simulation...")

        # generate the trajectory to go to qs
        trajectory_to_qs = self.get_traj_from_ruckig(q0, np.zeros(7), np.zeros(7),
                                                     self.qs, self.qs_dot, self.qs_dotdot,
                                                    margin_velocity=0.2, margin_acceleration=0.1)

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
        p.disconnect()

    def run(self):
        # ------------ Control Loop ------------ #
        dT = 5e-3
        rate = rospy.Rate(1.0 / dT)
        while not rospy.is_shutdown():
            # Publish state and fsm for debug
            self.fsm_state_pub.publish(self.fsm_state)

            # Check tracking error before throw
            if self.fsm_state == "THROWING" or self.fsm_state == "RELEASE":
                if (self.time_throw - rospy.get_time()) < 10e-3:
                    self.tracking_error_pos.append(
                        np.array(self.target_state.position) - np.array(self.robot_state.position))
                    self.tracking_error_vel.append(
                        np.array(self.target_state.velocity) - np.array(self.robot_state.velocity))
                    self.joint_velo_his.append(np.array(self.robot_state.velocity))

            ## ---------------- FSM ---------------- ##
            # # Setup or reset variables prior to throwing ##
            # if self.fsm_state == "WAIT_SIGNAL":
            #     continue

            if self.fsm_state == "IDLE":
                pdb.set_trace(header="PDB PAUSE: Press C to start homing...")
                print("HOMING...")
                current_robot_state = rospy.wait_for_message("/iiwa/joint_states", JointState)
                self.q0 = np.array(current_robot_state.position)
                self.homing_traj = self.get_traj_from_ruckig(self.q0, np.zeros(3), np.zeros(3),
                                                             self.qs, self.qs_dot, self.qs_dotdot,
                                                        margin_velocity=0.2, margin_acceleration=0.1)
                print("IDLING: initial state", self.q0, "homing trajectory duration", self.homing_traj.duration)

                # update state
                self.fsm_state = "HOMING"
                self.time_start_homing = rospy.get_time()

            elif self.fsm_state == "HOMING":

                # Activate integrator term when close to target
                error_position = np.array(self.robot_state.position) - np.array(self.qs)
                if np.linalg.norm(error_position) < self.ERROR_THRESHOLD:
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
                # release gripper
                # if time_now - self.time_start_throwing > self.throwing_traj.duration - self.GRIPPER_DELAY:
                #     threading.Thread(target=self.deactivate_gripper).start()

                if time_now - self.time_start_throwing > self.throwing_traj.duration - dT:
                    self.fsm_state = "SLOWING"
                    self.time_start_slowing = time_now
                    q_cur = np.array(self.robot_state.position)
                    qdot_cur = np.array(self.robot_state.velocity)
                    self.trajectory_back = self.get_traj_from_ruckig(q_cur, qdot_cur, self.qd_dotdot,
                                                                     self.qs, self.qs_dot, self.qs_dotdot,
                                                                margin_velocity=self.MARGIN_VELOCITY,
                                                                margin_acceleration=self.MARGIN_ACCELERATION * 0.5)
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
            # compute new trajectory to throw from current position
            q_cur = np.array(self.robot_state.position)
            qdot_cur = np.array(self.robot_state.velocity)
            self.throwing_traj = self.get_traj_from_ruckig(q_cur, qdot_cur, np.zeros(3),
                                                           self.qd, self.qd_dot, self.qd_dotdot,
                                                      margin_velocity=self.MARGIN_VELOCITY,
                                                      margin_acceleration=self.MARGIN_ACCELERATION)
            # inspect the throwing trajectory
            # self.simulate_trajectory(self.throwing_traj)
            # pdb.set_trace(header="PDB PAUSE: Press C to throw...")
            self.time_start_throwing = rospy.get_time()
            self.fsm_state = "THROWING"
            print("Throwing...")

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
        trajectory = Trajectory(3)
        _ = otg.calculate(inp, trajectory)
        return trajectory


if __name__ == '__main__':
    throwing_controller = Throwing_controller()

    for nTry in range(100):
        print("test number", nTry + 1)

        throwing_controller.fsm_state = "IDLE"
        throwing_controller.run()

        # Stop controller when ROS is stopped
        if rospy.is_shutdown():
            break