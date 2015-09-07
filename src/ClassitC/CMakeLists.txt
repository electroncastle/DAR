cmake_minimum_required(VERSION 2.8)
project( ClassitC )

#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "/home/jiri/Lake/HAR/opencv/build-fast/share/OpenCV/"  )
#set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/home/jiri/Lake/HAR/opencv/build-fast/share/OpenCV/" )
#set(OpenCV_DIR "/home/jiri/Lake/HAR/opencv/build-fast/")

# QT
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/home/jiri/Qt/5.5/gcc_64/lib/cmake/Qt5" )
  # Tell CMake to run moc when necessary:
  set(CMAKE_AUTOMOC ON)
  # As moc files are generated in the binary dir, tell CMake
  # to always look for includes there:
  set(CMAKE_INCLUDE_CURRENT_DIR ON)

FIND_PACKAGE(Qt5 REQUIRED Core)

find_package( OpenCV 2.4 REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

add_executable( ClassitC classification.cpp )

target_link_libraries( ClassitC ${OpenCV_LIBS})
target_link_libraries( ClassitC ${QT_LIBRARIES})

qt5_use_modules(ClassitC Core)

# MPI
#include_directories( "/home/jiri/Lake/HAR/lib/mpi/include" )
#target_link_libraries( extractFlow -L/home/jiri/Lake/HAR/lib/mpi/lib  -lmpi_cxx -lmpi -lopen-pal -lopen-rte )

# Caffe

include_directories( /usr/local/cuda/include/  ${CMAKE_CURRENT_SOURCE_DIR}/../caffe/build_release_mpi/include ${CMAKE_CURRENT_SOURCE_DIR}/../caffe/include)
target_link_libraries( ClassitC -L${CMAKE_CURRENT_SOURCE_DIR}/../caffe/build_release_mpi/lib/ -lcaffe-d -lproto-d -lz -lrt -lglog -lprotobuf -pthread)

