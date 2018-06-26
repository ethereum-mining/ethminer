# ethminer

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg)](https://github.com/RichardLitt/standard-readme)
[![Gitter](https://img.shields.io/gitter/room/nwjs/nw.js.svg)][Gitter]
[![Releases](https://img.shields.io/github/downloads/ethereum-mining/ethminer/total.svg)][Releases]
[![Coverity Scan Build Status](https://img.shields.io/coverity/scan/16024.svg)](https://scan.coverity.com/projects/ethereum-mining-ethminer-ac1b0eaf-5bed-4c82-bbab-a303c07bb6a7)

> Ethereum miner with OpenCL, CUDA and stratum support

**Ethminer** is an Ethash GPU mining worker: with ethminer you can mine every coin which relies on an Ethash Proof of Work thus including Ethereum, Ethereum Classic, Metaverse, Musicoin, Ellaism, Pirl, Expanse and others. This is the actively maintained version of ethminer. It originates from [cpp-ethereum] project (where GPU mining has been discontinued) and builds on the improvements made in [Genoil's fork]. See [FAQ](#faq) for more details.

## Features

* OpenCL mining
* Nvidia CUDA mining
* realistic benchmarking against arbitrary epoch/DAG/blocknumber
* on-GPU DAG generation (no more DAG files on disk)
* stratum mining without proxy
* OpenCL devices picking
* farm failover (getwork + stratum)


## Table of Contents

* [Install](#install)
* [Usage](#usage)
    * [Examples connecting to pools](#examples-connecting-to-pools)
* [Build](#build)
    * [Continuous Integration and development builds](#continuous-integration-and-development-builds)
    * [Building from source](#building-from-source)
* [Maintainers & Authors](#maintainers--authors)
* [Contribute](#contribute)
* [F.A.Q.](#faq)


## Install

[![Releases](https://img.shields.io/github/downloads/ethereum-mining/ethminer/total.svg)][Releases]

Standalone **executables** for _Linux_, _macOS_ and _Windows_ are provided in
the [Releases] section.
Download an archive for your operating system and unpack the content to a place
accessible from command line. The ethminer is ready to go.

| Builds | Release | Date |
| ------ | ------- | ---- |
| Last   | [![GitHub release](https://img.shields.io/github/release/ethereum-mining/ethminer/all.svg)](https://github.com/ethereum-mining/ethminer/releases) | [![GitHub Release Date](https://img.shields.io/github/release-date-pre/ethereum-mining/ethminer.svg)](https://github.com/ethereum-mining/ethminer/releases) |
| Stable | [![GitHub release](https://img.shields.io/github/release/ethereum-mining/ethminer.svg)](https://github.com/ethereum-mining/ethminer/releases) | [![GitHub Release Date](https://img.shields.io/github/release-date/ethereum-mining/ethminer.svg)](https://github.com/ethereum-mining/ethminer/releases) |


## Usage

The **ethminer** is a command line program. This means you launch it either
from a Windows command prompt or Linux console, or create shortcuts to
predefined command lines using a Linux Bash script or Windows batch/cmd file.
For a full list of available command, please run:

```sh
ethminer --help
```

### Examples connecting to pools

Check our [samples](docs/POOL_EXAMPLES_ETH.md) to see how to connect to different pools.

## Build

### Continuous Integration and development builds

| CI            | OS            | Status  | Development builds |
| ------------- | ------------- | -----   | -----------------  |
| [Travis CI]   | Linux, macOS  | [![Travis CI](https://img.shields.io/travis/ethereum-mining/ethminer.svg)][Travis CI]    | ✗ No build artifacts, [Amazon S3 is needed] for this |
| [AppVeyor]    | Windows       | [![AppVeyor](https://img.shields.io/appveyor/ci/ethereum-mining/ethminer.svg)][AppVeyor] | ✓ Build artifacts available for all PRs and branches |

The AppVeyor system automatically builds a Windows .exe for every commit. The latest version is always available [on the landing page](https://ci.appveyor.com/project/ethereum-mining/ethminer) or you can [browse the history](https://ci.appveyor.com/project/ethereum-mining/ethminer/history) to access previous builds.

To download the .exe on a build under `JOB NAME` select `Configuration: Release`, choose `ARTIFACTS` then download the zip file.

### Building from source

See [docs/BUILD.md](docs/BUILD.md) for build/compilation details.

## Maintainers & Authors

[![Gitter](https://img.shields.io/gitter/room/ethereum-mining/ethminer.svg)][Gitter]

The list of current and past maintainers, authors and contributors to the ethminer project.
Ordered alphabetically. [Contributors statistics since 2015-08-20].

| Name                  | Contact                                                      |     |
| --------------------- | ------------------------------------------------------------ | --- |
| Andrea Lanfranchi     | [@AndreaLanfranchi](https://github.com/AndreaLanfranchi)     | ETH: 0xa7e593bde6b5900262cf94e4d75fb040f7ff4727 |
| EoD                   | [@EoD](https://github.com/EoD)                               |     |
| Genoil                | [@Genoil](https://github.com/Genoil)                         |     |
| goobur                | [@goobur](https://github.com/goobur)                         |     |
| Marius van der Wijden | [@MariusVanDerWijden](https://github.com/MariusVanDerWijden) | ETH: 0x57d22b967c9dc64e5577f37edf1514c2d8985099 |
| Paweł Bylica          | [@chfast](https://github.com/chfast)                         | ETH: 0x8FB24C5b5a75887b429d886DBb57fd053D4CF3a2 |
| Philipp Andreas       | [@smurfy](https://github.com/smurfy)                         |     |
| Stefan Oberhumer      | [@StefanOberhumer](https://github.com/StefanOberhumer)       |     |


## Contribute

[![Gitter](https://img.shields.io/gitter/room/ethereum-mining/ethminer.svg)][Gitter]

To meet the community, ask general questions and chat about ethminer join [the ethminer channel on Gitter][Gitter].

All bug reports, pull requests and code reviews are very much welcome.


## License

Licensed under the [GNU General Public License, Version 3](LICENSE).


## F.A.Q

### Why is my hashrate with Nvidia cards on Windows 10 so low?

The new WDDM 2.x driver on Windows 10 uses a different way of addressing the GPU. This is good for a lot of things, but not for ETH mining.

* For Kepler GPUs: I actually don't know. Please let me know what works best for good old Kepler.
* For Maxwell 1 GPUs: Unfortunately the issue is a bit more serious on the GTX750Ti, already causing suboptimal performance on Win7 and Linux. Apparently about 4MH/s can still be reached on Linux, which, depending on ETH price, could still be profitable, considering the relatively low power draw.
* For Maxwell 2 GPUs: There is a way of mining ETH at Win7/8/Linux speeds on Win10, by downgrading the GPU driver to a Win7 one (350.12 recommended) and using a build that was created using CUDA 6.5.
* For Pascal GPUs: You have to use the latest WDDM 2.1 compatible drivers in combination with Windows 10 Anniversary edition in order to get the full potential of your Pascal GPU.

### Why is a GTX 1080 slower than a GTX 1070?

Because of the GDDR5X memory, which can't be fully utilized for ETH mining (yet).

### Are AMD cards also affected by slowdowns with increasing DAG size?

Only GCN 1.0 GPUs (78x0, 79x0, 270, 280), but in a different way. You'll see that on each new epoch (30K blocks), the hashrate will go down a little bit.

### Can I still mine ETH with my 2GB GPU?

Not really, your VRAM must be above the DAG size (Currently about 2.15 GB.) to get best performance. Without it severe hash loss will occur.

### What are the optimal launch parameters?

The default parameters are fine in most scenario's (CUDA). For OpenCL it varies a bit more. Just play around with the numbers and use powers of 2. GPU's like powers of 2.

### What does the `--cuda-parallel-hash` flag do?

[@davilizh](https://github.com/davilizh) made improvements to the CUDA kernel hashing process and added this flag to allow changing the number of tasks it runs in parallel. These improvements were optimised for GTX 1060 GPUs which saw a large increase in hashrate, GTX 1070 and GTX 1080/Ti GPUs saw some, but less, improvement. The default value is 4 (which does not need to be set with the flag) and in most cases this will provide the best performance.

### What is ethminer's relationship with [Genoil's fork]?

[Genoil's fork] was the original source of this version, but as Genoil is no longer consistently maintaining that fork it became almost impossible for developers to get new code merged there. In the interests of progressing development without waiting for reviews this fork should be considered the active one and Genoil's as legacy code.

### Can I CPU Mine?

No, use geth, the go program made for ethereum by ethereum.

### CUDA GPU order changes sometimes. What can I do?

There is an environment var `CUDA_DEVICE_ORDER` which tells the Nvidia CUDA driver how to enumerates the graphic cards.
The following values are valid:

* `FASTEST_FIRST` (Default) - causes CUDA to guess which device is fastest using a simple heuristic.
* `PCI_BUS_ID` - orders devices by PCI bus ID in ascending order.

To prevent some unwanted changes in the order of your CUDA devices you **might set the environment variable to `PCI_BUS_ID`**.
This can be done with one of the 2 ways:

* Linux:
    * Adapt the `/etc/environment` file and add a line `CUDA_DEVICE_ORDER=PCI_BUS_ID`
    * Adapt your start script launching ethminer and add a line `export CUDA_DEVICE_ORDER=PCI_BUS_ID`

* Windows:
    * Adapt your environment using the control panel (just search `setting environment windows control panel` using your favorite search engine)
    * Adapt your start (.bat) file launching ethminer and add a line `set CUDA_DEVICE_ORDER=PCI_BUS_ID` or `setx CUDA_DEVICE_ORDER PCI_BUS_ID`. For more info about `set` see [here](https://docs.microsoft.com/en-us/windows-server/administration/windows-commands/set_1), for more info about `setx` see [here](https://docs.microsoft.com/en-us/windows-server/administration/windows-commands/setx)

10. Insufficient CUDA driver

    ```shell
    Error: Insufficient CUDA driver: 9010
    ```

    You have to upgrade your Nvidia drivers. On Linux, install `nvidia-396` package or newer.


[Amazon S3 is needed]: https://docs.travis-ci.com/user/uploading-artifacts/
[AppVeyor]: https://ci.appveyor.com/project/ethereum-mining/ethminer
[cpp-ethereum]: https://github.com/ethereum/cpp-ethereum
[Contributors statistics since 2015-08-20]: https://github.com/ethereum-mining/ethminer/graphs/contributors?from=2015-08-20
[Genoil's fork]: https://github.com/Genoil/cpp-ethereum
[Gitter]: https://gitter.im/ethereum-mining/ethminer
[Releases]: https://github.com/ethereum-mining/ethminer/releases
[Travis CI]: https://travis-ci.org/ethereum-mining/ethminer
