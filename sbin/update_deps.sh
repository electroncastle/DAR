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


# Install Viper GT viewer 
cd ${DAR_ROOT}
mkdir -p opt
cd opt
wget http://prdownloads.sourceforge.net/viper-toolkit/viper-light-20050525.zip
unzip viper-light-20050525.zip
rm viper-light-20050525.zip
cd ${DAR_ROOT}
