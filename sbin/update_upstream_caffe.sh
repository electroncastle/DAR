#!/bin/bash

#
#   @file    update_upstream_caffe.sh
#   @author  Jiri Fajtl, <ok1zjf@gmail.com>
#   @date    13/7/2015
#   @version 0.1
# 
#   @brief Updates the forked Caffe branch from the upstream master
# 	Note the upstream branch must be added before manually
#	$ git remote add upstream https://github.com/BVLC/caffe.git
#	Check with:
#	$ git remote show upstream
#

SCRIPT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
DAR_PATH=$(dirname $SCRIPT_DIR)

git fetch upstream
git merge upstream/master
git push



