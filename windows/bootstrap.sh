#!/bin/bash
 
# Setup script for building Ethereum using Visual Studio Express 2013.
# Execute once in directory only containing cpp-ethereum
# Prerequisites:
#  - Visual Studio Express 2013 for Desktop
#  - On PATH: git, git-svn, wget, 7z

# stop on errors
set -e

# fetch CryptoPP-5.6.2
git svn clone -r 541:541 svn://svn.code.sf.net/p/cryptopp/code/trunk/c5 cryptopp

# fetch MiniUPnP-1.8
git clone git@github.com:miniupnp/miniupnp.git
cd miniupnp
git checkout tags/miniupnpd_1_8
cd ..

# fetch LevelDB (windows branch)
git clone https://code.google.com/p/leveldb/
cd leveldb
git checkout origin/windows
cd ..

# fetch and unpack boost-1.55
wget -O boost_1_55_0.7z http://sourceforge.net/projects/boost/files/boost/1.55.0/boost_1_55_0.7z/download
7z x boost_1_55_0.7z
mv boost_1_55_0 boost

# compile boost for x86 and x64
cd boost
cmd /c bootstrap.bat
./b2 --build-type=complete link=static runtime-link=static variant=debug,release threading=multi stage
mv stage/lib stage/Win32
./b2 --build-type=complete link=static runtime-link=static variant=debug,release threading=multi address-model=64 stage
mv stage/lib stage/x64
cd ..
