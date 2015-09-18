#!/bin/bash


#
#   @file    build_caffe.sh
#   @author  Jiri Fajtl, <ok1zjf@gmail.com>
#   @date    13/7/2015
#   @version 0.1
# 
#   @brief Builds & installs Caffe. 
#	This must be executed in a build subdirectory in the Caffe root dir
#	An old caffe installation must be removed first by calling 
#	remove_caffe_install.sh in the build directory of the currently installed Caffe
#	library!
# 
#

#SCRIPT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
#DAR_PATH=$(dirname $SCRIPT_DIR)
#echo $DAR_PATH

if [ "$1" == "debug" ]; then
	echo "Building DEBUG version"
	echo ${DAR_ROOT}
	cmake .. -DCMAKE_INSTALL_PREFIX:PATH=${DAR_ROOT} -DCMAKE_BUILD_TYPE=debug
else
	cmake .. -DCMAKE_INSTALL_PREFIX:PATH=${DAR_ROOT}
fi
make -j 8
make -j 8 pycaffe
make install



