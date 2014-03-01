@echo off
rem @file compileqt.bat
rem @author Tim Hughes <tim@twistedfury.com>
rem @date 2014
echo on

rem : import VC environment
call "%VS120COMNTOOLS%\VsDevCmd.bat"

rem : build for x64
msbuild /maxcpucount /p:Configuration=Release;Platform=x64 Ethereum.sln

rem : build for Win32
msbuild /maxcpucount /p:Configuration=Release;Platform=Win32 Ethereum.sln
