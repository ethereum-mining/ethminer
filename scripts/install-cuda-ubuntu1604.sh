#!/usr/bin/env bash
#
# Install the core CUDA_VER toolkit for a ubuntu-trusty (14.04) system. Requires the
# CUDA_VER environment variable to be set to the required version.
#
# Since this script updates environment variables, to execute correctly you must
# 'source' this script, rather than executing it in a sub-process.
#
# Taken from https://github.com/tmcdonell/travis-scripts.

set -e

export CUDA_VER=9.0.176-1
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_${CUDA_VER}_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604_${CUDA_VER}_amd64.deb
sudo apt-get update -qq
export CUDA_APT=${CUDA_VER:0:3}
export CUDA_APT=${CUDA_APT/./-}
sudo apt-get install -qy cuda-core-${CUDA_APT} cuda-cudart-dev-${CUDA_APT}
sudo apt-get clean
export CUDA_HOME=/usr/local/cuda-${CUDA_VER:0:3}
export PATH=${CUDA_HOME}/bin:${PATH}
