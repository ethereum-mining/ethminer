#!/bin/bash

# solves problem with macdeployqt on Qt 5.4 RC
# http://qt-project.org/forums/viewthread/50118

BUILD_FOLDER_PATH=$1
BUILD_QML_FOLDER_PATH="$BUILD_FOLDER_PATH/Resources/qml"
BUILD_PLUGINS_FOLDER_PATH="$BUILD_FOLDER_PATH/PlugIns"

if [ ! -d ${BUILD_QML_FOLDER_PATH} ]; then
    # we are not using any qml files
    # gracefully exit
    exit 0
fi

declare -a BROKEN_FILES;
k=0;
for j in $(find ${BUILD_QML_FOLDER_PATH} -name *.dylib); do
	BROKEN_FILES[${k}]=$j
	
	((k=k+1))
done


for i in "${BROKEN_FILES[@]}"; do
	REPLACE_STRING="$BUILD_FOLDER_PATH/"
	APP_CONTENT_FILE=${i//$REPLACE_STRING/""}
	IFS='/' read -a array <<< "$APP_CONTENT_FILE"
	LENGTH=${#array[@]}
	LAST_ITEM_INDEX=$((LENGTH-1))
	FILE=${array[${LENGTH} - 1]}
	
	ORIGINE_PATH=$(find ${BUILD_PLUGINS_FOLDER_PATH} -name ${FILE})
	ORIGINE_PATH=${ORIGINE_PATH//$REPLACE_STRING/""}
	s=""
	for((l=0;l<${LAST_ITEM_INDEX};l++)) do
		s=$s"../"
	done
	s=$s$ORIGINE_PATH
	echo "s: $s"
	
	REMOVE_BROKEN_ALIAS=$(rm -rf $i)
	RESULT=$(ln -s $s $i)
done

