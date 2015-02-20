# The Ethereum EVM JIT

EVM JIT is a library for just-in-time compilation of Ethereum EVM code.
It can be used to substitute classic interpreter-like EVM Virtual Machine in Ethereum client.

## Build

### Linux / Ubuntu

1. Install llvm-3.5-dev package
  1. For Ubuntu 14.04 using LLVM deb packages source: http://llvm.org/apt
  2. For Ubuntu 14.10 using Ubuntu packages
2. Build library with cmake
  1. `mkdir build && cd $_`
  2. `cmake .. && make`
3. Install library
  1. `sudo make install`
  2. `sudo ldconfig`
  
### OSX

1. Install llvm35
  1. `brew install llvm35 --disable-shared --HEAD`
2. Build library with cmake
  1. `mkdir build && cd $_`
  2. `cmake -DLLVM_DIR=/usr/local/lib/llvm-3.5/share/llvm/cmake .. && make`
3. Install library
  1. `make install` (with admin rights?)
  
### Windows

Ask me.

## Options

Options to evmjit library can be passed by environmental variables, e.g. `EVMJIT_CACHE=0 testeth --jit`.

Option        | Default value | Description
------------- | ------------- | ----------------------------------------------
EVMJIT_CACHE  | 1             | Enables on disk cache for compiled EVM objects
EVMJIT_DUMP   | 0             | Dumps generated LLVM module to standard output
  

