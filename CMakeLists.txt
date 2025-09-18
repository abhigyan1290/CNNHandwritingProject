cmake_minimum_required(VERSION 3.16)
project(mnist_cnn_eigen CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Try to find OpenCV; optional
find_package(PkgConfig)
if (PkgConfig_FOUND)
  pkg_check_modules(OPENCV opencv4)
endif()

add_executable(mnist_cnn
  src/train.cpp
  # headers are included via include dirs; no need to list them here
)

target_include_directories(mnist_cnn PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(mnist_cnn PRIVATE Eigen3::Eigen)

if (OPENCV_FOUND)
  target_compile_definitions(mnist_cnn PRIVATE OPENCV_OK)
  target_include_directories(mnist_cnn PRIVATE ${OPENCV_INCLUDE_DIRS})
  target_link_libraries(mnist_cnn PRIVATE ${OPENCV_LIBRARIES})
endif()
