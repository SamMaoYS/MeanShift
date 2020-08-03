cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(segmentation)

find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

include_directories(SYSTEM include)

file(GLOB resources "include/*.h" "src/*.cpp")

add_executable (main ${resources})

target_link_libraries (main ${OpenCV_LIBS} ${Boost_LIBS} stdc++fs)
