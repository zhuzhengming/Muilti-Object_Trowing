cmake_minimum_required(VERSION 3.0.2)
project(Multi-object_Throwing)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
        roscpp
        std_msgs
)


catkin_package(
        #  INCLUDE_DIRS include
        #  LIBRARIES path_planning
          CATKIN_DEPENDS std_msgs roscpp
        #  DEPENDS system_lib
)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
        # include
)

