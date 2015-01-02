# cpp-ethereum external dependencies

**This is Work-in-Progress!**

This directory hosts the external libraries that are needed to build cpp-ethereum.

To automatically download, build, and link libraries, do
```
cd extdep; mkdir build; cd build; cmake ..; make
```
this will take some time.


To check which libraries are already included, check `CMakeLists.txt`. Other libraries still need to be fetched via the system's package manager.

Libraries will be installed in `cpp-ethereum/extdep/install/<platform-name>`
