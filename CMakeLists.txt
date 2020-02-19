cmake_minimum_required(VERSION 3.10)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-10.0/bin/nvcc")

project(clothoidsG1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

FILE(GLOB clothoidsG1Lib "src/include/*.hh")
FILE(GLOB clothoidsG1Src "test/*.cc")

add_executable(clothoidsG1 ${clothoidsG1Src})