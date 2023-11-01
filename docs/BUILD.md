# Building from source

## Table of Contents

* [Requirements](#requirements)
    * [Common](#common)
    * [Linux](#linux)
        * [OpenCL support on Linux](#opencl-support-on-linux)
    * [macOS](#macos)
    * [Windows](#windows)
* [CMake configuration options](#cmake-configuration-options)
* [Disable Hunter](#disable-hunter)
* [Instructions](#instructions)
    * [Windows-specific script](#windows-specific-script)


## Requirements

This project uses [CMake] and [Hunter] package manager.

### Common

1. [CMake] >= 3.5
2. [Git](https://git-scm.com/downloads)
3. [Perl](https://www.perl.org/get.html), needed to build OpenSSL
4. [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) >= 9.0 (optional, install if you want NVidia CUDA support)

### Linux

1. GCC version >= 4.8, tested on GCC compiler 7.5.0
2. CMake version 3.21.5
``` shell
wget https://github.com/Kitware/CMake/releases/download/v3.21.5/cmake-3.21.5.tar.gz
tar -zxvf cmake-3.21.5.tar.gz
cd cmake-3.21.5
./bootstrap
make
sudo make install
```
3. DBUS development libs if building with `-DETHDBUS`. E.g. on Ubuntu run:

```shell
sudo apt install libdbus-1-dev
```

#### OpenCL support on Linux

If you're planning to use [OpenCL on Linux](https://github.com/ruslo/hunter/wiki/pkg.opencl#pitfalls)
you have to install the OpenGL libraries. E.g. on Ubuntu run:

```shell
sudo apt-get install mesa-common-dev
```

If you want to use locally installed [ROCm-OpenCL](https://rocmdocs.amd.com/en/latest/) package, use build flag `-DUSE_SYS_OPENCL=ON` with cmake config.

### macOS

1. GCC version >= TBF

### Windows

1. [Visual Studio 2017](https://www.visualstudio.com/downloads/); Community Edition works fine. **Make sure you install MSVC 2015 toolkit (v140).**

## Instructions

1. Make sure git submodules are up to date:

    ```shell
    git submodule update --init --recursive
    ```

2. Create a build directory:

    ```shell
    mkdir build
    cd build
    ```

3. Configure the project with CMake. Check out the additional [configuration options](#cmake-configuration-options).

    ```shell
    cmake ..
    ```

    **Note:** On Windows, it's possible to have issues with VS 2017 default compilers, due to CUDA expecting a specific toolset version; in that case, use the VS 2017 installer to get the VS 2015 compilers and pass the `-T v140` option:

    ```shell
    cmake .. -G "Visual Studio 15 2017 Win64"
    # or this if you have build errors in the CUDA step
    cmake .. -G "Visual Studio 15 2017 Win64" -T v140
    ```

4. Build the project using [CMake Build Tool Mode]. This is a portable variant of `make`.

    ```shell
    cmake --build .
    ```

    Note: On Windows, it is possible to have compiler issues if you don't specify the build config. In that case use:

    ```shell
    cmake --build . --config Release
    ```
5. Generate the executable
    ```shell
    make all
    ```
    will find the executable in **ethminer/ethminer** in the **build** directory
6. _(Optional, Linux only)_ Install the built executable:

    ```shell
    sudo make install
    ```

### Windows-specific script

Complete sample Windows batch file - **adapt it to your system**. Assumes that:

* it's placed one folder up from the ethminer source folder
* you have CMake installed
* you have Perl installed

```bat
@echo off
setlocal

rem add MSVC in PATH
call "%ProgramFiles(x86)%\Microsoft Visual Studio\2017\Community\Common7\Tools\VsMSBuildCmd.bat"

rem add Perl in PATH; it's needed for OpenSSL build
set "PERL_PATH=C:\Perl\perl\bin"
set "PATH=%PERL_PATH%;%PATH%"

rem switch to ethminer's source folder
cd "%~dp0\ethminer\"

if not exist "build\" mkdir "build\"

rem For CUDA 9.x pass also `-T v140`
cmake -G "Visual Studio 15 2017 Win64" -H. -Bbuild -DETHASHCL=ON -DETHASHCUDA=ON -DAPICORE=ON ..
cmake --build . --config Release --target package

endlocal
pause
```

## CMake configuration options

Pass these options to CMake configuration command, e.g.

```shell
cmake .. -DETHASHCUDA=ON -DETHASHCL=OFF
```

* `-DETHASHCL=ON` - enable OpenCL mining, `ON` by default.
* `-DETHASHCUDA=ON` - enable CUDA mining, `ON` by default.
* `-DAPICORE=ON` - enable API Server, `ON` by default.
* `-DBINKERN=ON` - install AMD binary kernels, `ON` by default.
* `-DETHDBUS=ON` - enable D-Bus support, `OFF` by default.
* `-DUSE_SYS_OPENCL=ON` - Use system OpenCL, `OFF` by default, unless on macOS. Specify to use local **ROCm-OpenCL** package.

## Disable Hunter

If you want to install dependencies yourself or use system package manager you can disable Hunter by adding
[`-DHUNTER_ENABLED=OFF`](https://docs.hunter.sh/en/latest/reference/user-variables.html#hunter-enabled)
to the configuration options.


[CMake]: https://cmake.org/
[CMake Build Tool Mode]: https://cmake.org/cmake/help/latest/manual/cmake.1.html#build-tool-mode
[Hunter]: https://docs.hunter.sh/
