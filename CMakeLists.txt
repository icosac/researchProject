cmake_minimum_required(VERSION 3.10)
#set(CMAKE_CUDA_COMPILER "/usr/local/cuda-10.0/bin/nvcc") #Sometimes one needs to set the path to the compiler

project(clothoidsG1 LANGUAGES CXX)
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-Wall -O3")

set(LIB libClothoids)
set(APP_EXEC clothoidsG1)

FILE(GLOB clothoidsG1Src "src/*.cc")
FILE(GLOB clothoidsG1Exec "test/*.cc")
ADD_LIBRARY(${LIB} STATIC ${clothoidsG1Src})

add_executable(${APP_EXEC} ${clothoidsG1Src} ${clothoidsG1Exec})

target_link_libraries(${APP_EXEC} ${LIB})
target_include_directories(${APP_EXEC} PRIVATE src/include/)