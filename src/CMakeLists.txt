cmake_minimum_required(VERSION 3.0.2)
project(Multi-object_Throwing)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages

find_package(catkin REQUIRED COMPONENTS
        rospy
        roscpp
        std_msgs
)

catkin_package(
        #  INCLUDE_DIRS include
          CATKIN_DEPENDS std_msgs roscpp rospy
        #  DEPENDS system_lib
)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
        # include
)

catkin_install_python(PROGRAMS
  scripts/trajectory_generation.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

