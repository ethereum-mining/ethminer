## ethminer-genoil

What is ethminer-0.9.41-genoil-1.x.x? 

Formerly known as Genoil's CUDA miner, ethminer-0.9.41-genoil-1.x.x is a fork of the stock ethminer version 0.9.41. While native CUDA support is its most significant difference, it has the following additional features:

- realistic benchmarking against arbitrary epoch/DAG/blocknumber
- custom DAG storage directory
- auto DAG directory cleanup
- stratum mining without proxy
- OpenCL devices picking
- farm failover (getwork + stratum)

### Usage

ethminer is a command line program. This means you launch it either from a Windows command prompt or Linux console, or create shortcuts to predefined command lines using a Linux Bash script or Windows batch/cmd file.
for a full list of available command, please run 

```
ethminer --help
```

### F.A.Q

1. Why is my hashrate with Nvidia cards on Windows 10 so low?
The new WDDM 2.0 driver on Windows 10 uses a different way of addressing the GPU. This is good for a lot of things, but not for ETH mining. There is a way of mining ETH at Win7/8/Linux speeds on Win10, by downgrading the GPU driver to a Win7 one (350.12 recommended) and using a [build that was created using CUDA 6.5](releases/cuda-6.5).

2. And what about the GTX750Ti?
Unfortunately the issue is a bit more serious on the GTX750Ti, already causing suboptimal performance on Win7 and Linux. Apparently about 5MH/s can still be reached on Linux, which, depending on ETH price, could still be profitable, considering the relatively low power draw.

3. Are AMD cards also affected by this issue?
Yes, but in a different way. While Nvidia cards have thresholds (i.e 2GB for 9x0 / Win7) of the DAG file size after which performance will drop steeply, on AMD cards the hashrate also drops with increasing DAG size, but more in a linear pattern. 

4. Can I still mine ETH with my 2GB GPU?
2GB should be sufficient for a while, altough it's become a bit uncertain if we'll stay below 2GB until the switch to PoS. I don't keep an exact list of all supported GPU's, but generally speaking the following cards should be ok:
AMD HD78xx, HD79xx, R9 2xx, R9 3xx, Fury.
Nvidia Geforce 6x0, 7x0, 8x0, 9x0, TITAN
Quadro, Tesla & FirePro's with similar silicon should be fine too.

5. Can I buy a private kernel from you that hashes faster?
No.

6. What are the optimal launch parameters?
The default parameters are fine in most scenario's (CUDA). For OpenCL it varies a bit more. Just play around with the numbers and use powers of 2. GPU's like powers of 2. 

7. Is your miner faster than the stock miner?
In CUDA yes, in OpenCL only on Nvidia .

### Building on Windows

- download or clone this repository
- download and install Visual Studio 12 2013 and CMake
- run [getstuff.bat](extdep/getstuff.bat) in [cpp-ethereum/extdep](extdep) 
- open a command prompt and navigate to cpp-ethereum directory

``` mkdir build 
cd build
cmake -DBUNDLE=cudaminer -G "Visual Studio 12 2013 Win64" ..
```

- if you don't want/need CUDA support, use "miner" instead of "cudaminer". This will only compile OpenCL support
- to speed up compilation a bit, you can add -DCOMPUTE=xx , where x is your CUDA GPU Compute version * 10. i.e -DCOMPUTE=52 for a GTX970.  
- you may disable stratum support by adding -DSTRATUM=0
- When CMake completes without errors, opn ethereum.sln created in the build directory in Visual Studio
- Set "ethminer" as startup project by right-clicking on it in the project pane
- Build. Run

### Building on Ubuntu

Note: this section was copied from [ethpool](https://ethpool.freshdesk.com/support/solutions/articles/8000032853-how-to-compile-genoils-cuda-miner-on-ubuntu)

Ubuntu 14.04. OpenCL only (for AMD cards)

```bash
sudo apt-get update
sudo apt-get -y install software-properties-common
add-apt-repository -y ppa:ethereum/ethereum
sudo apt-get update
sudo apt-get install git cmake libcryptopp-dev libleveldb-dev libjsoncpp-dev libjson-rpc-cpp-dev libboost-all-dev libgmp-dev libreadline-dev libcurl4-gnutls-dev ocl-icd-libopencl1 opencl-headers mesa-common-dev libmicrohttpd-dev build-essential -y
git clone https://github.com/Genoil/cpp-ethereum/
cd cpp-ethereum/
mkdir build
cd build
cmake -DBUNDLE=miner ..
make -j8
```

You can then find the executable in the ethminer subfolder

Ubuntu 14.04. OpenCL + CUDA (for NVIDIA cards)

```bash
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.5-18_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1404_7.5-18_amd64.deb
sudo apt-get -y install software-properties-common
sudo add-apt-repository -y ppa:ethereum/ethereum
sudo apt-get update
sudo apt-get install git cmake libcryptopp-dev libleveldb-dev libjsoncpp-dev libjson-rpc-cpp-dev libboost-all-dev libgmp-dev libreadline-dev libcurl4-gnutls-dev ocl-icd-libopencl1 opencl-headers mesa-common-dev libmicrohttpd-dev build-essential cuda -y
git clone https://github.com/Genoil/cpp-ethereum/
cd cpp-ethereum/
mkdir build
cd build
cmake -DBUNDLE=miner ..
make -j8
```

You can then find the executable in the ethminer subfolder