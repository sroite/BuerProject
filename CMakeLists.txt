cmake_minimum_required(VERSION 3.5.0)
project(buer)

# 找到Catkin和作为其组件的Choreonoid
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  trajectory_msgs
  choreonoid
)

# 包含Catkin的头文件路径 (会自动包含Choreonoid的)
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

# 定义包的依赖项
catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS
    roscpp
    rospy
    std_msgs
    message_runtime
)

set(CMAKE_CXX_STANDARD ${CHOREONOID_CXX_STANDARD})
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(${catkin_INCLUDE_DIRS})

# 将构建任务交给src目录下的CMakeLists.txt
add_subdirectory(controllers)