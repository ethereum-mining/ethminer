#!/bin/bash

CPP_ETHEREUM_PATH=$(pwd)
BUILD_DIR=$CPP_ETHEREUM_PATH/build
TEST_MODE=""

for i in "$@"
do
case $i in
	-builddir)
	shift
	((i++))
	BUILD_DIR=${!i}
	shift 
	;;
	--all)
	TEST_MODE="--all"
	shift 
	;;
	--filltests)
	TEST_FILL="--filltests"
	shift
	;;
esac
done

which $BUILD_DIR/test/testeth >/dev/null 2>&1
if [ $? != 0 ]
then
	echo "You need to compile and build ethereum with cmake -DPROFILING option to the build dir!"
	exit;
fi

OUTPUT_DIR=$BUILD_DIR/test/coverage
if which lcov >/dev/null; then
	if which genhtml >/dev/null; then
		echo Cleaning previous report...
		if [ -d "$OUTPUT_DIR" ]; then
			rm -r $OUTPUT_DIR
		fi
		mkdir $OUTPUT_DIR
		lcov --directory $BUILD_DIR --zerocounters
		lcov --capture --initial --directory $BUILD_DIR --output-file $OUTPUT_DIR/coverage_base.info

		echo Running testeth...
		$BUILD_DIR/test/testeth $TEST_MODE $TEST_FILL
		$BUILD_DIR/test/testeth -t StateTests --jit $TEST_MODE
		$BUILD_DIR/test/testeth -t VMTests --jit $TEST_MODE

		echo Prepearing coverage info...
		lcov --capture --directory $BUILD_DIR --output-file $OUTPUT_DIR/coverage_test.info
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
