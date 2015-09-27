#!/bin/bash

#
#   @file    setup.sh
#   @author  Jiri Fajtl, <ok1zjf@gmail.com>
#   @date    13/7/2015
#   @version 0.1
# 
#   @brief
# 
#   @section DESCRIPTION
#

if [ "$1" == "" ];
then
   echo "You must specify a manifest file with list of files TO PRESERVE !!!"
   exit
fi 

echo "Reading manifest from: "$1

for file in *;
do
  grep -q -F "$file" $1 
  if [ $? == 1 ]; then
     echo "Deleting: $file" 
     rm -rf "$file" 
  fi
done
