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
  
    - 用大一点的cube，贴四面
  
    - 在PC上创建一个新的object
  
    - 在lasa的github安装ros包
  
    - 修改名称和ip通过topic读取data
    
    - 数据频率大概在300hz
  
  
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

- (done)逻辑测试

- 轨迹跟踪测试，对每个关节进行最大速度的80%跟踪测试
  - 主要是要测试出gain和max_acceleration 和 max_jerk， max_velocity不能，因为投掷需要比较大的速度
  - 记录所有数据（当前位置，目标位置，当前速度，目标速度）还需要画出跟踪误差
    - 主要关注速度的跟踪
  
- ## 论文记录：

  - ### 底层控制实验

    - 位置控制
    - 关节控制
    - 阻抗控制
    - 正逆运动学控制
    - 高速轨迹跟踪
    
  - ### 理论知识和技术方案：
  
    - #### model-based:
    
      - 分为两个阶段：
        - 根据落点计算释放状态，但是需要考虑释放延迟和影响
        
          - velocity hedgehog generation:
            - mapping的key变为(phi, gamma, dis, z)，离线跑了8个小时
          
            - 针对fixed-based的改进，joint 0和joint 6都不贡献末端速度，第一个轴用于调整角度，最后一个轴用于调整hand的姿势
          - BRT：
            - 由ODE采样抛掷范围进行飞行轨迹计算得到需要的gamma, z。
            - brt_data: [z, dis, phi, gamma, layers, 5]->[r, z, r_dot, z_dot, v]
          - 匹配：
            - 方案一：
              - z对齐范围
              - dis < |AB|
              - r可达的
              - 需求的v可以满足的
              - r最小的，可以更加靠近目标物体
            - 方案二：
              -  COULD OPTIMIZE
        - 根据释放状态规划轨迹进行跟踪
        
      - 如果针对多物体投掷
        - 简化问题可以把灵巧手当作可以调整角度的两个gripper
          - target的区别太小了
          - hedgehog的生成考虑多加一个自由度的调整，也就是手指末端的相当于末端执行器的位置可以再动态调整
          - 归根结底是影响机器人的AE位置，以及速度
        - 异步投掷，解决的方案是进行两段motion planning
          - 可以扔的位置是相差远一点的
            - 方案一：
              - 搜索解的时候，去搜索q和q_dot尽可能相近的解然后进行两段规划并且组合
              - 组合的过程需要对两个目标点到达的先后顺序进行测试，哪一个需要的时间点更短
            - 方案二：
              - 遍历所有的轨迹组合，进行搜索耗时最小的
        
      - 实验：可以抛掷的距离最大大概在1.6m左右
        - 注意：手的安装方式要和仿真生成数据的方式一致
        
        - (done)先要重新修改urdf，也就是进行测试物体在手内的位置，增加一个固定关节来用mujoco推算
          - 可以通过mujoco直接计算得到与关节末端刚性连接的site的雅阁比矩阵
          
        - (done)测试release time
          - 论文 tube acceleration
          - 使用opti-trak，进行实验数据记录，对于实际落地进行记录，并分析release uncertainty 
          - 保持机器人在释放点之后依然在brt之内
          - 减少release的影响，给反向的加速度
          - 释放延时在40ms左右，提前多少个控制周期释放，需要测试
          
        - (done)末端执行器的朝向问题
          - 调整朝向出手的速度方向
        
        - 实验数据收集：
        
          - 轨迹误差：收集每个关节的目标和真实误差
        
        
          | 理想轨迹           | 通过搜出的目标的位置和速度进行计算   |
          | ------------------ | ------------------------------------ |
          | 释放瞬间计算的轨迹 | 机械臂跟踪到释放的位置和速度进行计算 |
          | 实际轨迹           | optitrack实际收集到的轨迹            |
        
          - 不同的trajectories
        
        - #### 实验问题记录：
        
          - (done)末端高度容易太多，撞桌子
            - 筛选z高度高的解
          - (done)需要提前进行一下仿真验证轨迹的可行性
            - 增加轨迹可视化功能
          - （done）给反向的加速度避免碰撞
          - （done）初始状态尽量利用joint0的速度
          - 选择不同phi角的解，收集不同phi角度的末端跟踪误差
            - phi角大，投的误差大，速度快，距离近
            - (done)phi角小，joint1没有太多的可以力矩加速，会和桌子碰撞
            - 选择joint1 target 速度小的解
            - joint 0, joint 1跟踪精度差，筛选不怎么需要他们的
              - (done)收集一个跟踪差的数据
              - 收集一个跟踪好的数据
                - 避免奇怪的姿势
          - brake过程是否需要：
            - posture1需要brake
              - 可以收集数据碰撞和不碰撞的视频
            - posture2不需要brake，所以可以用于结合投掷2个物体并且使用大的phi角度
          - 出手之后的飞行轨迹误差对比
          - 增加一个filter来选择使用使用哪些轴提供更大的速度
      
    - #### RL-based:
    
      - 好处在于可能能达到更大的投掷距离
      
      - 将整个过程解耦成2个部分，强化学习只决策释放时刻物体的状态，然后轨迹跟踪与强化学习无关。
        - **状态空间**：目标点和机器人的位置
          - 论文参考：
            - target position
            - 机器人状态
            - 投掷轨迹已经经过的时间，用于判断投掷轨迹的阶段
            - 历史action
        - **动作空间**：
          - 论文参考：
            - PD控制器的输出在关节空间下，改为输出增量
            - 释放信号采样：不是很好采样
            - 固定一定步数释放
        - **状态转移**：
          - model-based：理想状态，只有重力影响，飞行的动态方程
          - model-free：和mujoco交互，自己定义接口
        - **奖励函数**：
          - 接近程度
          - action scale
          - 落点速度大小
          - 参考论文：
            - 动作幅度
      
      - 训练：
        - stable-baselines3：PPO通过gym和mujoco模拟器交互
        - 需要自定义mujoco和gym环境的交互类：
          - 动作空间定义
          - 状态空间定义
          - 奖励函数定义
      
      - sim2real：
        - domain randomization：增加噪声
        - PD控制辨识
          - 利用real robot和mujoco来进行系统辨识对齐PD控制器
            - real robot进行正弦，脉冲响应记录
            - 利用MLP来mapping使得响应误差最小
      
      - 训练的step是150hz左右，接近iiwa的控制频率
      
      - 训练一直超过limit
      
        
      
    - #### 底层控制和跟踪
    
      - 基于优化的轨迹生成
      - 阻抗控制
        - 重力补偿
        - 正逆运动学
    
    - #### 数据分析
    
      - Velocity需要生成可视化可以看到有问题
        - panda max v: 4.3m/s
        - iiwa max v: 3.4m/s
      - distance也可视化出来了：
        - panda max dis: 2.0m
        - iiwa max dis: 1.7m
      
    - #### 问题记录：
    
      - (done)经常找不到解:
        - iiwa的速度太小了,末端生成的速度也小
        - brt的分辨率提高
        - 解决一：末端执行的位置是相对于机器人底座的
      - (done)自己训练出来的数据在panda的match过程是可以用的:
        - brt数据和hedgehog数据没有问题
        - 轨迹的筛选准则
      - (done)末端执行器的朝向：
        - 坐标转换
      - 利用灵巧手的抛掷问题
      - 强化学习搜索解的方案
      
    - ### 待办
    
      - (done)在mujoco里面生成在手指之间的末端hedgehog数据，并测试
      - 实物测试，使用optitrack
      - 测试release delay，参考tube acceleration
      - 考虑异步投掷两个物体的motion planning



### VLA抓取任务

- #### 理论：

  - 结构：
    - Visual encoder: DinoV2, SigLIP作为backbone, mapping图像到一些embeddings,
    - MLP projector: 把输出的embeddings mapping到大语言模型的输入
    - Llama2 7B: 输出一系列action token
    -  Action de-tokenizer: 把action token转化为可以输入给机器人的连续action

  - 训练：
    - 预训练：数据集Open X-Embodiment. 64 A100 15天
    - 微调：微调LoRA(大语言模型的低阶适应)效果最好，有接口了

- #### 实验：

  - 机器人6轴UR5，平台搭建和收集数据-2week
    - 就是RBG摄像头录制，操作机器人抓取机器人成功的视频，然后一帧一帧给机器人，大概100多组，单视角
    
    - 处理数据集-2week
      - 状态表示为：笛卡尔空间末端坐标6维+gripper开放状态1维度
      - **打包成RLDS数据格式**，强化学习用到的
    
  - 训练数据集
    - 使用1*4090，显存24G
      - 显存不够，利用mini-batch来叠加梯度实现小显存训练，不改变batch-size，分成小块来累加反向传播的梯度
        - grad_accumulation_steps
        - batch_size
      - 训练的frequency在5-10hz


- #### 微调

  - 之前只是输入RGB图像，增加机器人的关节状态和深度信息，使用单独的网络把其映射到与visual embedding同样的空间
  - 换用加入了action chunking的模型openvla-oft
    - 动作的连续性打包
  

