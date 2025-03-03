# Working Notes:

- #### Hardware:

  - **iiwa:** 

    ```
    roslaunch iiwa_driver iiwa_bringup.launch
    ```

    ![](/home/zhuzhengming/图片/iiwa.png)

    - PositionControl: 已知末端目标点的情况下，需要求解逆运用学
    - TorqueControl: 已知末端目标点的情况下，不需要求解逆运动学，只需要知道雅各比矩阵
    - 关节feedback:  average rate: 199.941hz  min: 0.002s max: 0.016s std dev: 0.00035s
  
  - **allegro**: 16 自由度，每个手指各4个
  
    - 暂时只有关节空间的控制
  
    ```
    roslaunch allegro_hand_controllers allegro_hand.launch HAND:=right
    ```
  
    ![](/home/zhuzhengming/图片/allegro.png)
  
    - ```
        NUM:=0|1|...
            ZEROS:=/path/to/zeros_file.yaml
            CONTROLLER:=grasp|pd|velsat|torque|sim
            RESPAWN:=true|false   Respawn controller if it dies.
            KEYBOARD:=true|false  (default is true)
            AUTO_CAN:=true|false  (default is true)
            CAN_DEVICE:=/dev/pcanusb1 | /dev/pcanusbNNN  (ls -l /dev/pcan* to see open CAN devices)
            VISUALIZE:=true|false  (Launch rviz)
            JSP_GUI:=true|false  (show the joint_state_publisher for *desired* joint angles)
      ```
  
  - **Opti-track**: 
  
    ```
    rosla vrpn_client_ros sample.launch
    ```
  
  - **Calibration**: controller_utils.py
  
    - ##### Position:
  
      - move_to_joints: 插值并运动目标关节到目标位置
      - _send_iiwa_position:发送位置控制指令
  
    - ##### Torque:
  
      - controller_utils2.py接受并且一直发送力矩控制
      - 通过函数iiwa_cartesion_impedance_control在笛卡尔空间下进行力矩控制

#### 控制模式：

- Cartesian space position control:
  - 可以不考虑重力补偿
- Cartesian space impedance control
  - 受到重力补偿的影响
  - 通过末端虚功和Jacobian矩阵反推关节指令
- Joint space impedance control:
  - 直接在关节空间进行PD阻抗控制



位置控制和阻抗控制有一个很大的区别是是否会对存在的误差不断积累输出量，可以看作是否有积分或者叠加项



### 工作记录：

##### 底层控制：

- (done)控制连续性问题,移动速度加大就断开连接:
  - 可能问题:运动不连续,可以画出加速度力矩等.
  - 解决:插值点太多,限制了速度

- (done)手抓取物体:
  - 采用位置伺服,预设控制

- (done)整合代码,给一个扔东西的代码

- (done)阻抗控制器对每个关节进行参数调节
  - 跟踪误差和速度
  - 跑一个sin轨迹

- (done)在joint space进行impedance control进行重力补偿，这里只有阻抗项和刚度项。
  
  - 重量为2kg，假设allegro的重心和末端中心重合
  
  $$
  tau = (kp * error_q + kd * error_{dq}) + \hat{J} F_{ext}
  $$
  
- 
  - J^理论上在最后一个轴有个offset


  - F_{ext}就是重力向下1X6的矩阵



#### 轨迹生成仿真

- 读取iiwa7 的数据，记得做一下sclae缩小

  | Axis | position scale | maximum torque | maximum velocity |
  | ---- | -------------- | -------------- | ---------------- |
  | A0   | +-170          | 176NM          | 98deg/s          |
  | A1   | +-120          | 176NM          | 98deg/s          |
  | A2   | +-170          | 110NM          | 100deg/s         |
  | A3   | +-120          | 110NM          | 130deg/s         |
  | A4   | +-170          | 110NM          | 140deg/s         |
  | A5   | +-120          | 40NM           | 180deg/s         |
  | A6   | +-175          | 40NM           | 180deg/s         |

  jerk数据可以先使用franka的constraint数据，因为iiwa性能更强：

  | jerk_max | 7500 | 3750 | 5000 | 6250 | 7500 | 10000 | 10000 |
  | -------- | ---- | ---- | ---- | ---- | ---- | ----- | ----- |

- (done)用FSM写一个系统逻辑

- (done)仿真环境mujoco下面测试跟踪效果

  - mujoco仿真环境下面存在控制器不适配问题。因为实体的机器人在启动的时候底层已经对机械臂进行了重力补偿了，但是仿真环境下还需要自己额外的补偿
  



#### 实体投掷测试：

- 逻辑测试
- 轨迹跟踪测试






- #### 论文记录：

  - ##### 底层控制

    - 位置控制
    - 关节控制
    - 阻抗控制
    - 正逆运动学控制

- 