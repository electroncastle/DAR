cmake_minimum_required(VERSION 2.8)
project( ofEval )

include(cmake/boost-python.cmake)

if (DEFINED OPENCV_ROOT)
    message("OPENCV_ROOT " ${OPENCV_ROOT})
else()
    set(OPENCV_ROOT "/home/jiri/Lake/DAR/share/OpenCV/"  )
    message("OPENCV_ROOT is NOT defined, setting " ${OPENCV_ROOT})
endif()



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
add_executable( ofEval
	ofEval.cpp 
	main.cpp 
	colorcode.cpp
	flowIO.cpp
        np_opencv_converter.cpp
        utils/conversion.cpp
)

# Python
find_package(PythonLibs 2.7 REQUIRED)
include_directories( ${PYTHON_INCLUDE_DIRS} )
target_link_libraries( ofEval ${PYTHON_LIBRARIES} boost_system boost_python)


#target_link_libraries( denseFlow_gpu ${OpenCV_LIBS} -lopencv_gpu)

#target_link_libraries( denseFlow_gpu ${OpenCV_LIBS})
target_link_libraries( ofEval ${OpenCV_LIBS})
#target_link_libraries( flowMaker ${QT_LIBRARIES})

qt5_use_modules(ofEval Core)

#include_directories( "/usr/include/mpi" )
#target_link_libraries( flowMaker -L/home/jiri/Lake/HAR/lib/mpi/lib  -lmpi_cxx -lmpi -lopen-pal -lopen-rte)
#target_link_libraries( flowMaker  -lmpi_cxx -lmpi -lopen-pal -lopen-rte )

boost_python_module(ofEval_module
        ofEval.cpp
        main.cpp
        colorcode.cpp
        flowIO.cpp
        np_opencv_converter.cpp
        utils/conversion.cpp
)
target_link_libraries( ofEval_module ${OpenCV_LIBS})
target_link_libraries( ofEval_module ${PYTHON_LIBRARIES} boost_system boost_python)

