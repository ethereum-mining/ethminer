#!/bin/bash

CPP_ETHEREUM_PATH=$(pwd)

which $CPP_ETHEREUM_PATH/build/test/testeth >/dev/null 2>&1
if [ $? != 0 ]
then
	echo "You need to compile and build ethereum with cmake -DPROFILING option to the build dir!"
	exit;
fi

OUTPUT_DIR="$CPP_ETHEREUM_PATH/build/test/coverage"
TESTETH=$CPP_ETHEREUM_PATH/build  #/test/CMakeFiles/testeth.dir

if which lcov >/dev/null; then
	if which genhtml >/dev/null; then
	lcov --directory $TESTETH --zerocounters
	echo Running testeth...
	$($CPP_ETHEREUM_PATH/build/test/testeth --all)
	$($CPP_ETHEREUM_PATH/build/test/testeth -t StateTests --jit --all)
	$($CPP_ETHEREUM_PATH/build/test/testeth -t VMTests --jit --all)
	echo Prepearing coverage info...
	else
	echo genhtml not found
	exit;
	fi
else
	echo lcov not found
	exit;
fi

if [ -d "$OUTPUT_DIR" ]; then
	echo Cleaning previous report...
	rm -r $OUTPUT_DIR
fi

mkdir $OUTPUT_DIR
lcov --capture --directory $TESTETH --output-file $OUTPUT_DIR/full_coverage.info
lcov --extract $OUTPUT_DIR/full_coverage.info *cpp-ethereum/* --output-file $OUTPUT_DIR/testeth_coverage.info
genhtml $OUTPUT_DIR/testeth_coverage.info --output-directory $OUTPUT_DIR/testeth

echo "Coverage info should be located at: $OUTPUT_DIR/testeth"
echo "Opening index..."

xdg-open $OUTPUT_DIR/testeth/index.html &
