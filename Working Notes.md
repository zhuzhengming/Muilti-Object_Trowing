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
        - **异步投掷**，解决的方案是进行两段motion planning
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
        
        - (done)单个物体实验数据收集：
        
          - 轨迹误差：收集每个关节的目标和真实误差
        
        
          | 理想轨迹           | 通过搜出的目标的位置和速度进行计算   |
          | ------------------ | ------------------------------------ |
          | 释放瞬间计算的轨迹 | 机械臂跟踪到释放的位置和速度进行计算 |
          | 实际轨迹           | optitrack实际收集到的轨迹            |
        
        - (done)误差分析（10次重复实验， 3个opti-track没捕捉到数据）：
        
          - tracking error：整个轨迹累计的**RMSE**，只考虑笛卡尔空间3维误差
            - pos: 0.05915，为什么这么小是因为只有最后一点的误差大
        
            - vel: 0.253131
        
          - release(grasping) error: 释放瞬间和飞行捕捉的瞬间的位置
            - 0.161423
        
          - real_actual:（caused by release）
            - 0.201554
        
          - real_target:
            - 0.131657
        
        | trajectory_pos_rmse | trajectory_vel_rmse | release_pos_rmse | real_target_rmse | real_actual_rmse |
        | ------------------- | ------------------- | ---------------- | ---------------- | ---------------- |
        | 0.059154            | 0.253131            | 0.161423         | 0.131657         | 0.201554         |
        
        - 图像：
        
          - (done)关节误差
          - (done)末端轨迹误差
          - (done)两条飞行轨迹对比
            - nominal 飞行轨迹落点不对，不画nominal trajectory了，直接可视化落点
          - (done)10次飞行轨迹对比
          - (done)释放延时:
            - 40-50ms
        
        - (done)不同的trajectories，视频
        
        - 设计一个实验去说明抛掷距离，抛射精度，碰撞概率之间的trade offer
        
        - #### 实验问题记录：
        
          - (done)末端高度容易太多，撞桌子
            - 筛选z高度高的解
          - (done)需要提前进行一下仿真验证轨迹的可行性
            - 增加轨迹可视化功能
          - （done）给反向的加速度避免碰撞
          - （done）初始状态尽量利用joint0的速度
          - (done)选择不同phi角的解，收集不同phi角度的末端跟踪误差
            - phi角大，投的误差大，速度快，距离近
            - (done)phi角小，joint1没有太多的可以力矩加速，会和桌子碰撞
            - 选择joint1 target 速度小的解
            - joint 0, joint 1跟踪精度差，筛选不怎么需要他们的
              - (done)收集一个跟踪差的数据
              - (done)收集一个跟踪好的数据
                - 避免奇怪的姿势
          - brake过程是否需要：
            - planner throw 需要
            - 实际分析不需要，会造成机械臂抖动，因为不怎么利用到planner throw
          - (done)出手之后的飞行轨迹误差对比
            - 数据分析记得统一末端位置是指尖
          - (done)增加一个filter来选择使用使用哪些轴提供更大的速度
          - (done)增加一个投掷泛化性测试，一个随机移动的盒子
          
        - #### 多物体抛掷实验记录
        
          - #### 问题记录：
        
            - greedy search函数的记录有问题，主要是generate_throw_config的函数需要知道qA, qA_dot，所以对于每一个qA，qA_dot的组合来遍历才对
          
            - 不用再跑一次hash map了 因为查询时间是不变，都是基于greedy search在跑，还需要重新跑一次greedy search仿真实验
          
          - #### comparison
          
            - naive search：
              - 给出怪异的抛掷方式，并且调大靠近物体的参数
            - greedy search：
              - 给出比较好的抛掷解，调大第一个抛掷的phi角度
          
          - #### Reactive 
          
            - 在greedy search or naive search的情况下进行 3 组位置的实验
          
      
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
        - PPO模型（on-policy）：
          - 为什么使用重要性采样：在重复利用旧的策略的时候，尽可能减少策略更新带来的偏差和方差问题
          - 可以减少策略更新的幅度，训练过程更加稳定
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
      
    - #### Model-based + Learning-based:
    
      - #### 实物实验设计：
    
        - 66小时离线生成dictionary
        - 对比：
          - naive search：直接跑两次各自最好的
          - greedy search：
            - 需要动态调整参数，第一段phi角度大，可以利用第一段的速度
            - 需要看看实验那些位置demo可以做
            - 设计释放顺序，减少碰撞可能，先释放posture 2的
          
          |               | computation time                                             | execution time                                               | offline calculation time |
          | ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------ |
          | naive search  | 0.0535<br />0.1065<br />0.0764<br />0.0698<br />0.0521<br />0.0456<br />0.0598<br />0.0839<br />0.0578<br />0.0428 | 6.581<br />3.909<br />4.420<br />4.871<br />6.274<br />6.021<br />4.282<br />4.782<br />5.393<br />5.421 | 0                        |
          | greedy search | 43.309<br />130.985<br />100.128<br />109.45<br />44.797<br />33.344<br />74.338<br />68.748<br />43.991<br />49.249 | 3.510<br />3.421<br />3.601<br />3.623<br />3.795<br />4.079<br />3.165<br />3.421<br />3.937<br />4.293 | 0                        |
          | hash map      | 0.0024<br />0.0055<br />0.0128<br />0.0102<br />0.0028<br />0.0036<br />0.0031<br />0.0025<br />0.0027<br />0.0030 | 同上                                                         | ~66h                     |
          
        - reactive throwing
          - 移动两个盒子来看效果
        
      - #### NN-model-based（diffusion model）:
      
        - 利用自己的filter来选出最佳的(q0, Boxes) -> target set
      
        - 然后利用监督学习来mapping减小在线寻找最佳解的时间
      
        - 特点是解的好坏取决于自己的filter rule
      
        - 记录sample数据的可利用率
      
        - 记录训练过程
      
      - #### RL-model(不好做):
      
        - 提前随机给定一个target box序列
        - **States:**
          - q0, box
        - **Observation**: 
          - 会生成一个(q_cur, box)的一个解集，也就是动作空间
          - 问题在于动作空间的内容大小是动态的，这个模型需要确定动作空间大小
          - 固定最大动作的数量
        - **actions**:
          - 学习一个概率分布，给出当前解集的一个determinastic的解
        - **rewards:**
          - current duration， whole duration
        - Maskable PPO
    
    
    
    
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
    - Visual encoder: DinoV2, SigLIP等transformer作为backbone, mapping图像到一些embeddings,
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
        - **grad_accumulation_steps**
        - batch_size
      - 训练的frequency在5-10hz
      - 输出的action限幅
      - 每次训练是2天多


- #### 微调

  - 之前只是输入RGB图像，增加机器人的关节状态和深度信息，使用单独的网络把其映射到与visual embedding同样的空间
  - 换用加入了action chunking的模型openvla-oft
    - 动作的连续性打包
  - HHA编码深度信息
    - 水平视差
    - 高于地面的高度
    - 像素的局部表面与推断重力方向的倾角
  
  - 自注意力：
    - 将输入嵌入到计算查询矩阵Q,键矩阵K,值矩阵V
    - **自注意力模型**的每个输出基于**全局信息**，可以很好的捕捉全局信息
    - 对于每个查询向量qi使用**键值对注意力机制**,得到注意力分布
  



















#### Speech notes:

My topic is ……, my supervisors are ……. I will go through these five parts. 

Traditional robotic throwing focus on **single-object task with gripper. or throwing multiple object in the same target position**. This research aims to define a unified planning framework for multi-object multi-target throwing

Timeline, …… (In laboratory nobody handles the throwing task using iiwa 7 and allegro)

The system framework is here, first we get the **target position** and robot base position via **optitrack**. Here are two **offline generated data structure** I'll illustrate later. Then searching and matching from them to get a lot of **feasible solutions**, but we need **filter** some solution according to some rules and choose a **final throwing state**. Next step is **trajectory generation and tracking**. I will illustrate them in details later.

Completed work, ……

Some **notations** are defined here,  A is robot base, B is target position, E is end effector. r is the distance from E to B in x0y plane, z is the height of object compare to robot base. phi is the yaw throwing angle, gamma is pitch throwing angel. 

The problem can be formulated as to **find configurations within joint constraint and inside Backwards reachable tube(BRT)**. BRT is a **continuous set**, which derives from object fly dynamics under **gravity** and can make sure the **landing state is inside target landing set**. Another offline generated data structure is a **dictionary with four keys**(height z, distance to base, and throwing angle gamma phi), the corresponding value is **maximum end effector velocity**. It is generated by solving a **linear program optimization problem.**  And here is some **visualization** of kinematic analysis. At height equals 0.15m, distance equals 0.7m, the distribution of maximum velocity is like this. And the **maximum Cartesian Velocity** is about 3.75m/s, **maximum throw distance** is around 1.5 m for iiwa 7.

After selecting a suitable solution, I applied **ruckig** to generate time optimal trajectory when **considering the limitations**. And then tracking the trajectory by **impedance controller in joint space.** 

We also need to consider  **grasping and release uncertainty** using allegro hand. To reduce these uncertainty, …… 

Next part is some experiment results. **Under predefined grasping posture,** the delay is around **40-50 ms**. Then here is a video about **reactive throwing** on real robot, I changed the target position randomly. It can alway generate a good trajectory and throw into the box as long as within the **reachable space**. 

And here is analysis of trajectory **tracking** in joint space, because accurate tracking ensures reach the **desire throwing state** and landing into target position.  For iiwa 7, as you can see, the **position and velocity error of joint 1** is large. I found that the **possible reason** it is that it has **little available torque to accelerate**. In torquecontrol mode, the maximum torque is bounded by 80 N·m whereas the datasheet says it can reach 175 N·m, and about 75 N·m is used to **compensate gravity** of robot. 

From the **comparison** of trajectory in **Cartesain space**, you can also find large gap in **z direction** caused by the joint 1.

Then I made a  **throwing object** like this to collection real flying data. For a **fixed target position**, the landing position has very small MAE and STD. 

This is a **quantitative analysis table**, as we can see, the **release state offset** is small after considering **release delay and grasping posture**. And the **main error is form the trajectory tracking.** Besides, the small standard deviation suggests the repeatability of this throwing motion plan. Finally, We can say that robot can throw into target box under **tolerance 0.15 m.** 

This part I want to talk some **trade offs** on **landing error, collision probability and throw distance**. For example, large target velocity of joint 1 **require longer trajectory** to accelerate, which increases the probability of **collision** with the table. large target velocity of joint 0 may cause **bigger landing error because of longer fly trajectory**. Neither joint 0 nor joint 1 lead to **short throwing distance.**

The future work mainly focus on Efficient planning for multi-object multi-target throwing, problem setup is given ……， goal is ……, The **Characteristics** of problem involves a trade-off between computation time and trajectory duration, here are two extremes, …… . Also, the throwing process can be described as a **sequential MDP** when considering trajectory duration time reward. One possible approach is Naive acceleration cone search, which is a **greedy strategy**, there is a trade off between computation time and execution time like we can **traverse much possible solution combinations and select the best one**. Another approach is reinforcement learning searching. 

Here is a video of naive search by **concatenating closest trajectory**. 

Summary，
