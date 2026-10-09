cmake_minimum_required(VERSION 3.22.1)
project(khronos_ros VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra)
endif()

option(BUILD_SHARED_LIBS "Build shared libs" ON)

find_package(ament_cmake REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(gflags REQUIRED)
find_package(hydra_ros REQUIRED)
find_package(khronos REQUIRED)
find_package(khronos_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# ##############################################################################
# Libraries #
# ##############################################################################

add_library(
  ${PROJECT_NAME}
  src/utils/ros_conversions.cpp
  src/experiments/experiment_logger.cpp
  src/experiments/experiment_manager.cpp
  src/experiments/experiment_directory.cpp
  src/khronos_pipeline.cpp
  src/visualization/active_window_visualizer.cpp
  src/visualization/dynamics_visualization.cpp
  src/visualization/visualization_utils.cpp
  src/visualization/spatio_temporal_visualizer.cpp)
target_include_directories(
  ${PROJECT_NAME}
  PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
         "$<INSTALL_INTERFACE:include>"
)
target_link_libraries(${PROJECT_NAME} PUBLIC khronos::khronos hydra_ros::hydra_ros PRIVATE cv_bridge::cv_bridge)
ament_target_dependencies(${PROJECT_NAME} PUBLIC khronos_msgs sensor_msgs)

# ##############################################################################
# Executables #
# ##############################################################################

add_executable(khronos_node app/khronos_node.cpp)
target_link_libraries(khronos_node ${PROJECT_NAME} ${gflags_LIBRARIES})

add_executable(spatio_temporal_visualizer_node app/spatio_temporal_visualizer_node.cpp)
target_link_libraries(spatio_temporal_visualizer_node ${PROJECT_NAME} ${gflags_LIBRARIES})

# ##############################################################################
# Export #
# ##############################################################################

install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}-targets ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)
install(TARGETS khronos_node spatio_temporal_visualizer_node RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include/)
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

# Install Python scripts
if(BUILD_TESTING)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  enable_testing()
  add_test(NAME evaluate_dynamics_bag
    COMMAND ${Python3_EXECUTABLE} -m unittest discover -s ${CMAKE_CURRENT_SOURCE_DIR}/tests -v)
  find_package(GTest REQUIRED)
  add_executable(test_dynamics_visualization tests/test_dynamics_visualization.cpp
    src/visualization/dynamics_visualization.cpp)
  target_include_directories(test_dynamics_visualization PRIVATE include)
  target_link_libraries(test_dynamics_visualization PRIVATE khronos::khronos GTest::gtest_main)
  ament_target_dependencies(test_dynamics_visualization PUBLIC sensor_msgs)
  add_test(NAME dynamics_visualization COMMAND test_dynamics_visualization)
endif()

install(PROGRAMS
  src/visualization/spatio_temporal_visualizer_gui.py
  scripts/odom_to_tf.py
  scripts/evaluate_dynamics_bag.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_targets(${PROJECT_NAME}-targets HAS_LIBRARY_TARGET)
ament_export_dependencies(gflags hydra_ros khronos khronos_msgs)
ament_package()
