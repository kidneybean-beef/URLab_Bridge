cmake_minimum_required(VERSION 3.16)
project(urlab_image_transport_relay)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rmw REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(msgpack-cxx REQUIRED)

pkg_check_modules(ZMQ REQUIRED libzmq)

add_library(urlab_camera_runtime
  src/runtime_description.cpp
)

target_include_directories(urlab_camera_runtime PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

if(TARGET msgpack-cxx)
  target_link_libraries(urlab_camera_runtime PUBLIC msgpack-cxx)
endif()

add_executable(image_transport_relay
  src/image_transport_relay.cpp
)

ament_target_dependencies(image_transport_relay
  rclcpp
  rmw
  sensor_msgs
  image_transport
)

add_executable(urlab_camera_bridge
  src/urlab_camera_bridge.cpp
)

target_include_directories(urlab_camera_bridge PRIVATE
  ${ZMQ_INCLUDE_DIRS}
)

target_compile_options(urlab_camera_bridge PRIVATE
  ${ZMQ_CFLAGS_OTHER}
)

target_link_directories(urlab_camera_bridge PRIVATE
  ${ZMQ_LIBRARY_DIRS}
)

target_link_libraries(urlab_camera_bridge
  urlab_camera_runtime
  ${ZMQ_LIBRARIES}
)

if(TARGET msgpack-cxx)
  target_link_libraries(urlab_camera_bridge msgpack-cxx)
endif()

ament_target_dependencies(urlab_camera_bridge
  rclcpp
  rmw
  sensor_msgs
  image_transport
)

install(TARGETS image_transport_relay urlab_camera_bridge
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_runtime_description
    test/test_runtime_description.cpp
  )
  target_link_libraries(test_runtime_description urlab_camera_runtime)
endif()

ament_package()
