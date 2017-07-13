# ethminer

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg)](https://github.com/RichardLitt/standard-readme)
[![Gitter](https://img.shields.io/gitter/room/nwjs/nw.js.svg)][Gitter]
[![Releases](https://img.shields.io/github/downloads/atom/atom/total.svg)][Releases]

> Ethereum miner with OpenCL, CUDA and stratum support

The ethminer is an Ethereum GPU mining worker. It origins in [cpp-ethereum]
project (where GPU mining has been discontinued). Then hugely improved in
[Genoil's fork].

### Features

- OpenCL mining
- Nvidia CUDA mining
- realistic benchmarking against arbitrary epoch/DAG/blocknumber
- on-GPU DAG generation (no more DAG files on disk)
- stratum mining without proxy
- OpenCL devices picking
- farm failover (getwork + stratum)


## Table of Contents

- [Install](#install)
- [Usage](#usage)
- [Build](#build)
  - [Continuous Integration and development builds](#continuous-integration-and-development-builds)
  - [Building from source](#building-from-source)
  - [CMake configuration options](#cmake-configuration-options)
- [Maintainer](#maintainer)  
- [Contribute](#contribute)
- [F.A.Q.](#faq)


## Install

[![Releases](https://img.shields.io/github/downloads/atom/atom/total.svg)][Releases]

Standalone **executables** for _Linux_, _macOS_ and _Windows_ are provided in
the [Releases] section.
Download an archive for your operating system and unpack the content to a place
accessible from command line. The ethminer is ready to go.


## Usage

The **ethminer** is a command line program. This means you launch it either
from a Windows command prompt or Linux console, or create shortcuts to
predefined command lines using a Linux Bash script or Windows batch/cmd file.
For a full list of available command, please run

```sh
ethminer --help
```


## Build

### Continuous Integration and development builds

| CI            | OS            | Status  | Development builds |
| ------------- | ------------- | -----   | -----------------  |
| [Travis CI]   | Linux, macOS  | [![Travis CI](https://img.shields.io/travis/ethereum-mining/ethminer.svg)][Travis CI]    | ✗ No build artifacts, [Amazon S3 is needed] for this |
| [AppVeyor]    | Windows       | [![AppVeyor](https://img.shields.io/appveyor/ci/ethereum-mining/ethminer.svg)][AppVeyor] | ✓ Build artifacts available for all PRs and branches |

The AppVeyor system automatically builds a Windows .exe for every commit. The latest version is always available [on the landing page](https://ci.appveyor.com/project/ethereum-mining/ethminer) or you can [browse the history](https://ci.appveyor.com/project/ethereum-mining/ethminer/history) to access previous builds.

To download the .exe on a build under 'JOB NAME' select 'Configuration: Release', choose 'ARTIFACTS' then download the zip file.


### Building from source

This project uses [CMake] and [Hunter] package manager.

1. Create a build directory.

   ```sh
   mkdir build; cd build
   ```

2. Configure the project with CMake. Check out additional
   [configuration options](#cmake-configuration-options).

   ```sh
   cmake ..
   ```

3. Build the project using [CMake Build Tool Mode]. This is a portable variant
   of `make`.

   ```sh
   cmake --build .
   ```

4. _(Optional, Linux only)_ Install the built executable.

   ```sh
   sudo make install
   ```

### CMake configuration options

Pass these options to CMake configuration command, e.g.

```sh
cmake .. -DETHASHCUDA=ON -DETHASHCL=OFF
```

- `-DETHASHCL=ON` - enable OpenCL mining, `ON` by default,
- `-DETHASHCUDA=ON` - enable CUDA mining, `OFF` by default,
- `-DETHSTRATUM=ON` - build with Stratum protocol support, `ON` by default.


## Maintainer

[![Gitter](https://img.shields.io/gitter/room/ethereum-mining/ethminer.svg)][Gitter]

- Paweł Bylica [@chfast](https://github.com/chfast)


## Contribute

[![Gitter](https://img.shields.io/gitter/room/ethereum-mining/ethminer.svg)][Gitter]

To meet the community join [the ethminer channel on Gitter][Gitter].

All bug reports, pull requests and code reviews are very much welcome.


## F.A.Q

1. Why is my hashrate with Nvidia cards on Windows 10 so low?

   The new WDDM 2.x driver on Windows 10 uses a different way of addressing the GPU. This is good for a lot of things, but not for ETH mining.
   For Kepler GPUs: I actually don't know. Please let me know what works best for good old Kepler.
   For Maxwell 1 GPUs: Unfortunately the issue is a bit more serious on the GTX750Ti, already causing suboptimal performance on Win7 and Linux. Apparently about 4MH/s can still be reached on Linux, which, depending on ETH price, could still be profitable, considering the relatively low power draw.
   For Maxwell 2 GPUs: There is a way of mining ETH at Win7/8/Linux speeds on Win10, by downgrading the GPU driver to a Win7 one (350.12 recommended) and using a build that was created using CUDA 6.5.
   For Pascal GPUs: You have to use the latest WDDM 2.1 compatible drivers in combination with Windows 10 Anniversary edition in order to get the full potential of your Pascal GPU.

2. Why is a GTX 1080 slower than a GTX 1070?

   Because of the GDDR5X memory, which can't be fully utilized for ETH mining (yet).

3. Are AMD cards also affected by slowdowns with increasing DAG size?

   Only GCN 1.0 GPUs (78x0, 79x0, 270, 280), but in a different way. You'll see that on each new epoch (30K blocks), the hashrate will go down a little bit.

4. Can I still mine ETH with my 2GB GPU?

   No.

5. What are the optimal launch parameters?

   The default parameters are fine in most scenario's (CUDA). For OpenCL it varies a bit more. Just play around with the numbers and use powers of 2. GPU's like powers of 2.


[Amazon S3 is needed]: https://docs.travis-ci.com/user/uploading-artifacts/
[AppVeyor]: https://ci.appveyor.com/project/ethereum-mining/ethminer
[CMake]: https://cmake.org
[CMake Build Tool Mode]: https://cmake.org/cmake/help/latest/manual/cmake.1.html#build-tool-mode
[cpp-ethereum]: https://github.com/ethereum/cpp-ethereum
[Genoil's fork]: https://github.com/Genoil/cpp-ethereum
[Gitter]: https://gitter.im/ethereum-mining/ethminer
[Hunter]: https://docs.hunter.sh
[Releases]: https://github.com/ethereum-mining/ethminer/releases
[Travis CI]: https://travis-ci.org/ethereum-mining/ethminer
