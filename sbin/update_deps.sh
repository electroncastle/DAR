#!/bin/bash


if [ ! -d "caffe" ]; then
	git clone https://github.com/electroncastle/caffe.git
else 
	git pull
fi

if [ ! -d "opencv" ]; then
	git clone https://github.com/Itseez/opencv.git
else
	git pull
fi

if [ ! -d "opencv_contrib" ]; then
	git clone https://github.com/Itseez/opencv_contrib.git
else
	git pull
fi

