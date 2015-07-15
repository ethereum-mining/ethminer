call "%VS90COMNTOOLS%/vsvars32.bat"

set BUILD_DIR=build
set BIN_DIR=bin

mkdir %BUILD_DIR%
cd %BUILD_DIR%
cmake -G "NMake Makefiles" ../
nmake
cd ..

