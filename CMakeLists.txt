cmake_minimum_required(VERSION 3.10)
#set(CMAKE_CUDA_COMPILER "/usr/local/cuda-10.0/bin/nvcc") #Sometimes one needs to set the path to the compiler
include(GoogleTest)

project(DynamicCurve LANGUAGES CXX)
set(LIBCC libDynamicCurveCC)
set(LIBCU libDynamicCurveCU)
set(APP_EXEC dynamicCurve)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-Wall -O3")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(SRCCC "srcCC")
set(SRCCCF "${SRCCC}/*.cc")
set(SRCINC src/include)

IF(${CUDA_ON} MATCHES ON)
  enable_language(CUDA)
  file(GLOB dynamicCurveExec "exec/main.cu")
  add_definitions(-DCUDA_ON)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++11 -O3 -arch=sm_75 -DCUDA_ON --default-stream per-thread --compiler-options -Wall")
  set(CMAKE_CUDA_ARCHITECTURES 75)
  set(CUDA_SEPARABLE_COMPILATION ON)
  set(SRCCU "srcCU")
  set(SRCCUF "${SRCCU}/*.cu")
  message(STATUS "Compiling CUDA with: ${CMAKE_CUDA_FLAGS}")
ELSE()
  file(GLOB dynamicCurveExec "exec/main.cc")
ENDIF()

message(STATUS "Compiling C++ with: -std=c++${CMAKE_CXX_STANDARD} ${CMAKE_CXX_FLAGS}")

file(GLOB dynamicCurveSrcCC ${SRCCCF})
file(GLOB dynamicCurveSrcCU ${SRCCUF})

message(STATUS "${dynamicCurveSrcCC}")
message(STATUS "${dynamicCurveSrcCU}")

include_directories(${SRCCC}/include include)
add_library(${LIBCC} STATIC ${dynamicCurveSrcCC})
IF(${CUDA_ON} MATCHES ON)
  include_directories(${SRCCU}/include include)
  add_library(${LIBCU} STATIC ${dynamicCurveSrcCU})
ENDIF()

#add_executable(${APP_SRC} ${dynamicCurveSrc})
add_executable(${APP_EXEC} ${dynamicCurveExec})

target_include_directories(${APP_EXEC} PRIVATE ${SRCINC})

IF(${CUDA_ON} MATCHES ON)
  target_link_libraries(${APP_EXEC} ${LIBCC} ${LIBCU})
ELSE()
 target_link_libraries(${APP_EXEC} ${LIBCC})
ENDIF()

#add_subdirectory(test)

IF(${TEST} MATCHES "GTEST")
  #GTEST_TEST
  add_definitions(-DGTEST)
  add_subdirectory(ExternalLibs/googletest)
  enable_testing()

  add_executable(ConfigurationTest test/configurationTest.cc)
  add_executable(ClothoidG1TestCC test/clothoidG1TestCC.cc)
  add_executable(DubinsTestCC test/dubinsTestCC.cc)
  IF(${CUDA_ON} MATCHES ON)
    #add_executable(ClothoidG1TestCU test/clothoidG1TestCU.cc)
  ENDIF()
  
  target_link_libraries(ConfigurationTest gtest gtest_main)
  target_link_libraries(DubinsTestCC gtest gtest_main ${LIBCC})
  target_link_libraries(ClothoidG1TestCC gtest gtest_main ${LIBCC})
  target_include_directories(ConfigurationTest PRIVATE ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
  target_include_directories(DubinsTestCC PRIVATE ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
  target_include_directories(ClothoidG1TestCC PRIVATE ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
  IF(${CUDA_ON} MATCHES ON)
    #target_link_libraries(ClothoidG1TestCU gtest gtest_main)
    #target_include_directories(ClothoidG1TestCU PRIVATE ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
  ENDIF()
  
  gtest_discover_tests(ConfigurationTest)
  gtest_discover_tests(DubinsTestCC)
  gtest_discover_tests(ClothoidG1TestCC)
  IF(${CUDA_ON} MATCHES ON)
    #gtest_discover_tests(ClothoidG1TestCU)
  ENDIF()

ELSEIF(${TEST} MATCHES "BOOST")
  #BOOST_TEST
  add_definitions(-DBOOST)
  add_definitions(-DBOOST_TEST_DYN_LINK)
  
  enable_testing()
  find_package(Boost COMPONENTS system filesystem unit_test_framework REQUIRED)
  set(Boost_USE_MULTITHREADED TRUE)
  
  add_executable(ConfigurationTest test/configurationTest.cc)
  add_executable(ClothoidG1TestCC test/clothoidG1TestCC.cc)
  add_executable(DubinsTestCC test/dubinsTestCC.cc)
  IF(${CUDA_ON} MATCHES ON)
    #add_executable(DubinsTestCU test/dubinsTestCU.cc)
  ENDIF()
  
  target_link_libraries(ConfigurationTest ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
  target_link_libraries(ClothoidG1TestCC ${LIBCC} ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
  target_link_libraries(DubinsTestCC ${LIBCC} ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
  IF(${CUDA_ON} MATCHES ON)
    #target_link_libraries(DubinsTestCU ${LIBCU} ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
  ENDIF()

  target_include_directories(ConfigurationTest PRIVATE ${SRCINC})
  target_include_directories(ClothoidG1TestCC PRIVATE ${SRCINC})
  target_include_directories(DubinsTestCC PRIVATE ${SRCINC})

  add_test(NAME ConfigurationTest COMMAND ConfigurationTest)
  add_test(NAME ClothoidG1TestCC COMMAND ClothoidG1TestCC)
  add_test(NAME DubinsTestCC COMMAND DubinsTestCC)
  IF(${CUDA_ON} MATCHES ON)
    #add_test(NAME DubinsTestCU COMMAND DubinsTestCU)
  ENDIF()
ENDIF()
