cmake_minimum_required(VERSION 2.8)
project(GSLIC)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

set(GSLIC_LIB
slic_gpu.h
slic.h
slic_gpu.cu
)

list(APPEND "-std=c++11 -ftree-vectorize")

cuda_add_library(gslic_lib
						${GSLIC_LIB}
						OPTIONS -gencode arch=compute_35,code=compute_35
				)
target_link_libraries(gslic_lib ${CUDA_LIBRARY})

add_executable(main main.cpp)
target_link_libraries(main gslic_lib ${OpenCV_LIBS})