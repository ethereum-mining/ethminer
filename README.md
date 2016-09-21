## ethminer-genoil

What is ethminer-0.9.41-genoil-1.x.x? 

Formerly known as Genoil's CUDA miner, ethminer-0.9.41-genoil-1.x.x is a fork of the stock ethminer version 0.9.41. While native CUDA support is its most significant difference, it has the following additional features:

- realistic benchmarking against arbitrary epoch/DAG/blocknumber
- on-GPU DAG generation (no more DAG files on disk)
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
The new WDDM 2.x driver on Windows 10 uses a different way of addressing the GPU. This is good for a lot of things, but not for ETH mining. 
For Kepler GPUs: I actually don't know. Please let me know what works best for good old Kepler.
For Maxwell 1 GPUs: Unfortunately the issue is a bit more serious on the GTX750Ti, already causing suboptimal performance on Win7 and Linux. Apparently about 4MH/s can still be reached on Linux, which, depending on ETH price, could still be profitable, considering the relatively low power draw.
For Maxwell 2 GPUs: There is a way of mining ETH at Win7/8/Linux speeds on Win10, by downgrading the GPU driver to a Win7 one (350.12 recommended) and using a [build that was created using CUDA 6.5](releases/cuda-6.5).
For Pascal GPUs: You have to use the latest WDDM 2.1 compatible drivers in combination with Windows 10 Anniversary edition in order to get the full potential of your Pascval GPU.

2. Why is a GTX1080 slower than a GTX1070?
Because of the GDDR5X memory, which can't be fully utilized for ETH mining (yet).

3. Are AMD cards also affected by slowdowns with increasing DAG size?
Only GCN 1.0 GPUs (78x0, 79x0, 270, 280), but in a different way. You'll see that on each new epoch (30K blocks), the hashrate will go down a little bit.

4. Can I still mine ETH with my 2GB GPU?
2GB should be sufficient for a little while, altough you will have to set the following environment variables (Windows example, use export on Linux):

```
setx GPU_FORCE_64BIT_PTR 0
setx GPU_MAX_HEAP_SIZE 100
setx GPU_USE_SYNC_OBJECTS 1
setx GPU_SINGLE_ALLOC_PERCENT 100
setx GPU_MAX_ALLOC_PERCENT = 100
```

5. Can I buy a private kernel from you that hashes faster?
No.

6. What are the optimal launch parameters?
The default parameters are fine in most scenario's (CUDA). For OpenCL it varies a bit more. Just play around with the numbers and use powers of 2. GPU's like powers of 2. 

7. Is your miner faster than the stock miner?
In CUDA yes, in OpenCL only on Nvidia .

### Branches and versions

The master branch always contains the stable release. Currently that's 1.1.7. Then you may find branches like 110, 108, 120, whuich are either archives of previous major versions or beta releases of upcoming work.


### Releases

Windows x64 binaries can be found in the /releases folder.

### Building on Windows

- download or clone this repository
- download and install Visual Studio 12 2013 and CMake
- run [getstuff.bat](extdep/getstuff.bat) in [cpp-ethereum/extdep](extdep) 
- open a command prompt and navigate to cpp-ethereum directory

``` 
mkdir build 
cd build
cmake -DBUNDLE=cudaminer -G "Visual Studio 12 2013 Win64" ..
```

- if you don't want/need CUDA support, use "miner" instead of "cudaminer". This will only compile OpenCL support
- to speed up compilation a bit, you can add -DCOMPUTE=xx , where x is your CUDA GPU Compute version * 10. i.e -DCOMPUTE=52 for a GTX970.  
- you may disable stratum support by adding -DETH_STRATUM=0
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
cmake -DBUNDLE=cudaminer ..
make -j8
```

You can then find the executable in the ethminer subfolder