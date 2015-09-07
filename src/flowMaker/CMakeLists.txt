cmake_minimum_required(VERSION 2.8)
project( flowMaker )

set(OPENCV_ROOT "/home/jiri/Lake/DAR/share/OpenCV/"  )


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${OPENCV_ROOT}  )
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${OPENCV_ROOT} "/home/jiri/Qt/5.5/gcc_64/lib/cmake/Qt5" )
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${OPENCV_ROOT} )
#set(OpenCV_DIR "/home/jiri/Lake/HAR/opencv/build-fast/")

  # Tell CMake to run moc when necessary:
  set(CMAKE_AUTOMOC ON)
  # As moc files are generated in the binary dir, tell CMake
  # to always look for includes there:
  set(CMAKE_INCLUDE_CURRENT_DIR ON)

FIND_PACKAGE(Qt5 REQUIRED Core)
#message("QT inc " ${QT5_INCLUDE_DIR} )

find_package( OpenCV 3.0 REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
#include_directories( ${QT_INCLUDE_DIR} )
include_directories( ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

#add_executable( denseFlow_gpu denseFlow_gpu.cpp )
add_executable( flowMaker 
	extractFlow.cpp 
	imageProcessor.cpp
)

#target_link_libraries( denseFlow_gpu ${OpenCV_LIBS} -lopencv_gpu)

#target_link_libraries( denseFlow_gpu ${OpenCV_LIBS})
target_link_libraries( flowMaker ${OpenCV_LIBS})
#target_link_libraries( flowMaker ${QT_LIBRARIES})

qt5_use_modules(flowMaker Core)

include_directories( "/usr/include/mpi" )
#target_link_libraries( flowMaker -L/home/jiri/Lake/HAR/lib/mpi/lib  -lmpi_cxx -lmpi -lopen-pal -lopen-rte)
target_link_libraries( flowMaker  -lmpi_cxx -lmpi -lopen-pal -lopen-rte )
