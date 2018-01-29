#!/bin/bash
cd /root/ethNipper

MYHOST=`echo "$HOSTNAME"|awk '{print toupper($0)}'`
echo $MYHOST

#export GPU_FORCE_64BIT_PTR=0
export GPU_MAX_HEAP_SIZE=100
export GPU_USE_SYNC_OBJECTS=1
export GPU_MAX_ALLOC_PERCENT=100
export GPU_SINGLE_ALLOC_PERCENT=100
./ethNipper --opencl-platform 0 --cl-kernel 3 --farm-recheck 2000 -RH -SP 1 -HWMON -G -S eth-eu1.nanopool.org:9999 -O 0xd78a188af99a9a95b46c3ab5adaa7474ee8e0d36.$MYHOST/ml@sntt.de:x


