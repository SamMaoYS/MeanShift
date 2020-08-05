cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(segmentation)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

include_directories(SYSTEM include)

file(GLOB resources "include/*.h" "src/*.cpp")

add_executable (main ${resources})

if(OpenMP_CXX_FOUND)
    target_link_libraries (main PUBLIC OpenMP::OpenMP_CXX ${OpenCV_LIBS} stdc++fs)
endif()
