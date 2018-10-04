@echo off

setlocal

pushd %~dp0

if not defined COVDIR set "COVDIR=C:\cov-analysis"
if defined COVDIR if not exist "%COVDIR%" (
  echo.
  echo ERROR: Coverity not found in "%COVDIR%"
  goto End
)

call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\Common7\Tools\VsMSBuildCmd.bat"


:Cleanup
if exist "..\build"      rd /q /s "..\build"
if exist "..\cov-int"    rd /q /s "..\cov-int"
if exist "ethminer.lzma" del "ethminer.lzma"
if exist "ethminer.tar"  del "ethminer.tar"
if exist "ethminer.tgz"  del "ethminer.tgz"


:Main
set "PERL_PATH=C:\Perl\perl\bin"
set "PATH=%PERL_PATH%;%PATH%"

mkdir ..\build\
pushd ..\build\

cmake -G "Visual Studio 15 2017 Win64" -H. -Bbuild -T v140,host=x64 -DETHASHCL=ON -DETHASHCUDA=ON -DAPICORE=ON ..
"%COVDIR%\bin\cov-build.exe" --dir "..\cov-int" cmake --build . --config Release --target package
popd


:tar
set PATH=C:\MSYS\bin;%PATH%
tar --version 1>&2 2>nul || (echo. & echo ERROR: tar not found & goto SevenZip)
tar caf "ethminer.lzma" "..\cov-int"
goto End


:SevenZip
call :SubDetectSevenzipPath

rem Coverity is totally bogus with lzma...
rem And since I cannot replicate the arguments with 7-Zip, just use tar/gzip.
if exist "%SEVENZIP%" (
  "%SEVENZIP%" a -ttar "ethminer.tar" "..\cov-int"
  "%SEVENZIP%" a -tgzip "ethminer.tgz" "ethminer.tar"
  if exist "ethminer.tar" del "ethminer.tar"
  goto End
)


:SubDetectSevenzipPath
for %%G in (7z.exe) do (set "SEVENZIP_PATH=%%~$PATH:G")
if exist "%SEVENZIP_PATH%" (set "SEVENZIP=%SEVENZIP_PATH%" & exit /B)

for %%G in (7za.exe) do (set "SEVENZIP_PATH=%%~$PATH:G")
if exist "%SEVENZIP_PATH%" (set "SEVENZIP=%SEVENZIP_PATH%" & exit /B)

for /f "tokens=2*" %%A in (
  'reg QUERY "HKLM\SOFTWARE\7-Zip" /v "Path" 2^>nul ^| find "REG_SZ" ^|^|
   reg QUERY "HKLM\SOFTWARE\Wow6432Node\7-Zip" /v "Path" 2^>nul ^| find "REG_SZ"') do set "SEVENZIP=%%B\7z.exe"
exit /B


:End
popd
echo. & echo Press any key to close this window...
pause >nul
endlocal
exit /b
