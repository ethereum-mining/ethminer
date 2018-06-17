# Building from source

## General

This project uses [CMake] and [Hunter] package manager.

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

    Note: On Windows, it's possible to have issues with VS 2017 default compilers; in that case, use the VS 2017 installer to get the VS 2015 compilers and use:

    ```shell
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

    Complete sample Windows batch file - **adapt it to your system**. Assumes that:

    * it's placed one folder up from the ethminer source folder
    * you have CMake installed
    * you have VS 2017 with the 140 toolset installed; this is needed because CUDA doesn't compile with the 141 toolset currently
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

    cmake -G "Visual Studio 15 2017 Win64" -H. -Bbuild -T v140 -DETHASHCL=ON -DETHASHCUDA=ON -DAPICORE=ON ..
    cmake --build . --config Release --target package

    endlocal
    pause
    ```

5. _(Optional, Linux only)_ Install the built executable:

    ```shell
    sudo make install
    ```

## OpenCL support on Linux

If you're planning to use [OpenCL on Linux](https://github.com/ruslo/hunter/wiki/pkg.opencl#pitfalls)
you have to install the OpenGL libraries. E.g. on Ubuntu run:

```shell
sudo apt-get install mesa-common-dev
```

## Disable Hunter

If you want to install dependencies yourself or use system package manager
you can disable Hunter by adding
[`-DHUNTER_ENABLED=OFF`](https://docs.hunter.sh/en/latest/reference/user-variables.html#hunter-enabled)
to configuration options.

## CMake configuration options

Pass these options to CMake configuration command, e.g.

```shell
cmake .. -DETHASHCUDA=ON -DETHASHCL=OFF
```

* `-DETHASHCL=ON` - enable OpenCL mining, `ON` by default,
* `-DETHASHCUDA=ON` - enable CUDA mining, `ON` by default.
* `-DAPICORE=ON` - enable API Server, `ON` by default.
* `-DETHDBUS=ON` - enable D-Bus support, `OFF` by default.


[CMake]: https://cmake.org/
[CMake Build Tool Mode]: https://cmake.org/cmake/help/latest/manual/cmake.1.html#build-tool-mode
[Hunter]: https://docs.hunter.sh/
