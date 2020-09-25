cmake_minimum_required(VERSION 3.10)
#set(CMAKE_CUDA_COMPILER "/usr/local/cuda-10.0/bin/nvcc") #Sometimes one needs to set the path to the compiler
include(GoogleTest)

project(DynamicCurve LANGUAGES CXX)
#enable_language(CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-Wall -O3")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(LIB libDynamicCurve)
set(APP_EXEC dynamicCurve)

include_directories(src/include)

file(GLOB dynamicCurveSrc "src/*.cc")
file(GLOB dynamicCurveExec "exec/main.cc")

add_library(${LIB} STATIC ${dynamicCurveSrc})

add_executable(${APP_EXEC} ${dynamicCurveSrc} ${dynamicCurveExec})

target_link_libraries(${APP_EXEC} ${LIB})
target_include_directories(${APP_EXEC} PRIVATE src/include/)

#add_subdirectory(test)

IF(${TEST} MATCHES "GTEST")
    #GTEST_TEST
    add_definitions(-DGTEST)
    add_subdirectory(ExternalLibs/googletest)
    enable_testing()

    add_executable(ClothoidG1Test test/clothoidG1Test.cc)
    add_executable(ConfigurationTest test/configurationTest.cc)

    target_link_libraries(ConfigurationTest gtest gtest_main)
    target_link_libraries(ClothoidG1Test gtest gtest_main ${LIB})
    target_include_directories(ConfigurationTest PRIVATE ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
    target_include_directories(ClothoidG1Test PRIVATE ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

    gtest_discover_tests(ConfigurationTest)
    gtest_discover_tests(ClothoidG1Test)

ELSEIF(${TEST} MATCHES "BOOST")
    #BOOST_TEST
    add_definitions(-DBOOST)
    enable_testing()
    find_package(Boost COMPONENTS system filesystem unit_test_framework REQUIRED)
    set(Boost_USE_MULTITHREADED TRUE)

    add_definitions(-DBOOST_TEST_DYN_LINK)
    
    add_executable(ConfigurationTest test/configurationTest.cc)
    add_executable(ClothoidG1Test test/clothoidG1Test.cc)
    target_link_libraries(ConfigurationTest ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
    target_link_libraries(ClothoidG1Test ${LIB} ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})

    add_test(NAME ConfigurationTest COMMAND ConfigurationTest)
    add_test(NAME ClothoidG1Test COMMAND ClothoidG1Test)
ENDIF()
