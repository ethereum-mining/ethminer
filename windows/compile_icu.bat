@echo off
rem @file compile_icu.bat
rem @author Tim Hughes <tim@twistedfury.com>
rem @date 2014
echo on

rem : import VC environment
call "%VS120COMNTOOLS%\VsDevCmd.bat"

rem : build for platform
msbuild /maxcpucount /p:Configuration=Release;Platform=%1% source/allinone/icu.sln
