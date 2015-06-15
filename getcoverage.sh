#!/bin/bash

CPP_ETHEREUM_PATH=$(pwd)

if [ ! -d "$CPP_ETHEREUM_PATH/build/test" ]; then
   echo "You need to compile and build ethereum with cmake -DPROFILING option to the build dir!"
   exit;	 
fi

OUTPUT_DIR="$CPP_ETHEREUM_PATH/build/test/coverage"
TESTETH=$CPP_ETHEREUM_PATH/build  #/test/CMakeFiles/testeth.dir

if which lcov >/dev/null; then
    if which genhtml >/dev/null; then
	lcov --directory $TESTETH --zerocounters
	echo Running testeth...
	$($CPP_ETHEREUM_PATH/build/test/testeth)
	echo Prepearing coverage info...
    else
	echo genhtml not found
	exit;
    fi
else
    echo lcov not found
    exit;
fi

echo Cleaning previous report...
rm -r $OUTPUT_DIR/testeth
rm $OUTPUT_DIR/full_coverage.info
rm $OUTPUT_DIR/testeth_coverage.info
lcov --capture --directory $TESTETH --output-file $OUTPUT_DIR/full_coverage.info
lcov --extract $OUTPUT_DIR/full_coverage.info *cpp-ethereum/* --output-file $OUTPUT_DIR/testeth_coverage.info
genhtml $OUTPUT_DIR/testeth_coverage.info --output-directory $OUTPUT_DIR/testeth

echo "Coverage info should be located at: $CPP_ETHEREUM_PATH/build/test/coverage/testeth"
echo "Opening index..."

xdg-open $CPP_ETHEREUM_PATH/build/test/coverage/testeth/index.html &
