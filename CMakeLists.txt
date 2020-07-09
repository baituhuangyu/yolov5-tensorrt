cmake_minimum_required(VERSION 2.6)

project(yolov5)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11;-g;-G;-gencode;arch=compute_30;code=sm_30)

include_directories(${PROJECT_SOURCE_DIR}/include)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message("embed_platform on")
    include_directories(/usr/local/cuda/targets/aarch64-linux/include)
    link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
else()
    message("embed_platform off")
    include_directories(/usr/local/cuda/include)
    link_directories(/usr/local/cuda/lib64)
    include_directories(/mnt/data/yu.huang/work/TensorRT-6.0.1.5/include)
    link_directories(/mnt/data/yu.huang/work/TensorRT-6.0.1.5/lib)
endif()


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

cuda_add_library(yololayer SHARED ${PROJECT_SOURCE_DIR}/yololayer.cu)

find_package(OpenCV)
include_directories(OpenCV_INCLUDE_DIRS)
include_directories(/usr/local/include/opencv4)

#add_library(common SHARED
#        ${PROJECT_SOURCE_DIR}/common.hpp
#        ${PROJECT_SOURCE_DIR}/logging.h
#        ${PROJECT_SOURCE_DIR}/utils.h
#        ${PROJECT_SOURCE_DIR}/yololayer.cu
#        ${PROJECT_SOURCE_DIR}/yololayer.h
#        ${PROJECT_SOURCE_DIR}/yolov5s.cpp
#        )


add_executable(yolov5s ${PROJECT_SOURCE_DIR}/yolov5s.cpp)
target_link_libraries(yolov5s nvinfer)
target_link_libraries(yolov5s cudart)
target_link_libraries(yolov5s yololayer)
target_link_libraries(yolov5s ${OpenCV_LIBS})

#add_executable(openVideo ${PROJECT_SOURCE_DIR}/openVideo.cpp)
#target_link_libraries(openVideo ${OpenCV_LIBS})


add_definitions(-O2 -pthread)

