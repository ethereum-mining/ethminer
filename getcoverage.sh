#!/bin/bash

CPP_ETHEREUM_PATH=$(pwd)

which $CPP_ETHEREUM_PATH/build/test/testeth >/dev/null 2>&1
if [ $? != 0 ]
then
	echo "You need to compile and build ethereum with cmake -DPROFILING option to the build dir!"
	exit;
fi

OUTPUT_DIR="$CPP_ETHEREUM_PATH/build/test/coverage"
TESTETH=$CPP_ETHEREUM_PATH/build

if which lcov >/dev/null; then
	if which genhtml >/dev/null; then
		echo Cleaning previous report...
		if [ -d "$OUTPUT_DIR" ]; then
			rm -r $OUTPUT_DIR
		fi
		mkdir $OUTPUT_DIR
		lcov --directory $TESTETH --zerocounters
		lcov --capture --initial --directory $TESTETH --output-file $OUTPUT_DIR/coverage_base.info

		echo Running testeth...
		$CPP_ETHEREUM_PATH/build/test/testeth --all
		$CPP_ETHEREUM_PATH/build/test/testeth -t StateTests --jit --all
		$CPP_ETHEREUM_PATH/build/test/testeth -t VMTests --jit --all

		echo Prepearing coverage info...
		lcov --capture --directory $TESTETH --output-file $OUTPUT_DIR/coverage_test.info
		lcov --add-tracefile $OUTPUT_DIR/coverage_base.info --add-tracefile $OUTPUT_DIR/coverage_test.info --output-file $OUTPUT_DIR/coverage_all.info
		lcov --extract $OUTPUT_DIR/coverage_all.info *cpp-ethereum/* --output-file $OUTPUT_DIR/coverage_export.info
		genhtml $OUTPUT_DIR/coverage_export.info --output-directory $OUTPUT_DIR/testeth		
	else
		echo genhtml not found
		exit;
	fi
else
	echo lcov not found
	exit;
fi

echo "Coverage info should be located at: $OUTPUT_DIR/testeth"
echo "Opening index..."

xdg-open $OUTPUT_DIR/testeth/index.html &
