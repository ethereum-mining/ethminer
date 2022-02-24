#!/bin/sh
#
# Install the core CUDA_VER toolkit for Ubuntu 16.04.
# Requires the CUDA_VER environment variable to be set to the required version.
#
# Since this script updates environment variables, to execute correctly you must
# 'source' this script, rather than executing it in a sub-process.
#
# Taken from https://github.com/tmcdonell/travis-scripts.

set -e

CUDA_VER=9.1.85-1
if [ "$1" != "" ]; then
  CUDA_VER=$1
fi
if [ "$CUDA_VER" = "8" ]; then
  CUDA_VER=8.0.61-1
  CUDA_PACKAGE=cuda-core
elif [ "$CUDA_VER" = "9" ]; then
  CUDA_VER=9.1.85-1
elif [ "$CUDA_VER" = "9.1" ]; then
  CUDA_VER=9.1.85-1
elif [ "$CUDA_VER" = "9.2" ]; then
  CUDA_VER=9.2.148-1
elif [ "$CUDA_VER" = "10" ]; then
  CUDA_VER=10.0.130-1
elif [ "$CUDA_VER" = "10.1" ]; then
  CUDA_VER=10.2.243-1
elif [ "$CUDA_VER" = "10.2" ]; then
  CUDA_VER=10.2.89-1
elif [ "$CUDA_VER" = "11" ]; then
  CUDA_VER=11.2.67-1
elif [ "$CUDA_VER" = "11.1" ]; then
  CUDA_VER=11.1.105-1
elif [ "$CUDA_VER" = "11.2" ]; then
  CUDA_VER=11.2.67-1
fi

if [ -z $CUDA_PACKAGE ]; then
  CUDA_PACKAGE=cuda-nvcc
fi

sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
CUDA_APT=$(echo $CUDA_VER | sed 's/\.[0-9]\+\-[0-9]\+$//;s/\./-/')
if [ $CUDA_VER -lt 11 ]; then
  wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_${CUDA_VER}_amd64.deb
  sudo dpkg -i cuda-repo-ubuntu1604_${CUDA_VER}_amd64.deb
else
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-ubuntu1604.pin
  sudo mv cuda-ubuntu1604.pin /etc/apt/preferences.d/cuda-repository-pin-600
  sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/ /"
fi
sudo apt-get update -qq
sudo apt-get install -qy $CUDA_PACKAGE-$CUDA_APT cuda-cudart-dev-$CUDA_APT
sudo apt-get clean
CUDA_APT=$(echo $CUDA_APT | sed 's/-/./')
CUDA_HOME=/usr/local/cuda-$CUDA_APT
PATH=${CUDA_HOME}/bin:${PATH}
export CUDA_HOME
export PATH
