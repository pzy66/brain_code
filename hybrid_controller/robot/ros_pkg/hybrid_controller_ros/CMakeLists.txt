cmake_minimum_required(VERSION 3.0.2)
project(hybrid_controller_ros)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  rospy
  std_msgs
  std_srvs
)

add_message_files(
  FILES
  CylindricalTeleop.msg
  RobotState.msg
)

add_service_files(
  FILES
  MoveCyl.srv
  MoveCylAuto.srv
  PickCyl.srv
  PickWorld.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  std_srvs
)

catkin_package(
  CATKIN_DEPENDS message_runtime rospy std_msgs std_srvs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
