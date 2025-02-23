#!/bin/zsh

gnome-terminal --tab --title "ROS Controller" -- zsh -c "
  roslaunch Multi-object_Throwing allegro_iiwa_driver.launch hand:=left model:=7 controller:=TorqueController;
  exec zsh" 


sleep 1 


gnome-terminal --tab --title "Torque Service" -- zsh -c "
  conda init zsh
  sleep 1
  conda activate iiwa;
  export PYTHONPATH=\"/opt/ros/noetic/lib/python3/dist-packages:/home/zhuzhengming/workspace/Object_throwing/Muilti-Object_Trowing/scripts:\$PYTHONPATH\";
  export LD_LIBRARY_PATH=\"/opt/ros/noetic/lib:\$LD_LIBRARY_PATH\";
 # python /home/zhuzhengming/workspace/Object_throwing/Muilti-Object_Trowing/utils/torque_service.py;
  exec zsh" 
