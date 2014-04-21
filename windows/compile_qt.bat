@echo off
rem @file compileqt.bat
rem @author Tim Hughes <tim@twistedfury.com>
rem @date 2014

rem : enable use prefix if we want to produce standalone Qt binaries
rem : off by default since this takes longer and duplicates all the headers
set USE_PREFIX=0

rem : echo commands so we can see what's going on
echo on

rem : select platform and toolset from first argument
IF %1%==x64 (set PLATFORM=x64&set TOOLSET=x86_amd64&set) ELSE (set PLATFORM=Win32&set TOOLSET=x86=)

rem : import VC environment vars
call "%VS120COMNTOOLS%\..\..\VC\vcvarsall.bat" %TOOLSET%

rem : assume our root Qt dir is the current dir
set QT=%CD%

set PATH=%QT%\Src\gnuwin32\bin;%PATH%

rem : create the build folder and add the qtbase/bin folder to the PATH
if not exist %QT%\%PLATFORM% (
	set DO_CONFIGURE=1
	mkdir %QT%\%PLATFORM%
) else (
	set DO_CONFIGURE=0
)
if %USE_PREFIX%==1 (
	if not exist %QT%\%PLATFORM%-Build mkdir %QT%\%PLATFORM%-Build
	if not exist %QT%\%PLATFORM%\qtbase mkdir %QT%\%PLATFORM%\qtbase
	cd %QT%\%PLATFORM%-Build
	set QT_PREFIX=-prefix %Qt%\%PLATFORM%\qtbase
	set QT_TARGETS=module-qtbase-install_subtargets module-qtquick1-install_subtargets module-qtwebkit-install_subtargets
) else (
	cd %QT%\%PLATFORM%
	set QT_PREFIX=
	set QT_TARGETS=module-qtbase module-qtquick1 module-qtwebkit
)
set PATH=%CD%\qtbase\bin;%PATH%

rem : run Qt configure with desired settings
if %DO_CONFIGURE%==1 (
	call %QT%\Src\configure.bat %QT_PREFIX% -opensource -confirm-license -debug-and-release -opengl desktop -platform win32-msvc2013 -icu -I "%QT%\..\icu\include" -L "%QT%\..\icu\lib_%PLATFORM%" -no-compile-examples -nomake tests -nomake examples -no-accessibility
)

rem : compile and install module-qtbase
%QT%\jom\jom %QT_TARGETS%
