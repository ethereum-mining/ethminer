#!/bin/bash

# solves problem with macdeployqt on Qt 5.4 RC and Qt 5.5
# http://qt-project.org/forums/viewthread/50118

BUILD_FOLDER_PATH=$1
BUILD_QML_FOLDER_PATH="$BUILD_FOLDER_PATH/Resources/qml"
BUILD_PLUGINS_FOLDER_PATH="$BUILD_FOLDER_PATH/PlugIns"

if [ -d ${BUILD_QML_FOLDER_PATH} ]; then

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
fi

# replace framework links 
declare -a BROKEN_FRAMEWORKS;
k=0;
BUILD_FRAMEWORKS_FOLDER_PATH="$BUILD_FOLDER_PATH/Frameworks"
for j in $(find ${BUILD_FRAMEWORKS_FOLDER_PATH} -name Qt*.framework); do
	BROKEN_FRAMEWORKS[${k}]=$j
	((k=k+1))
done
for i in "${BROKEN_FRAMEWORKS[@]}"; do
	FRAMEWORK_FILE=$i/$(basename -s ".framework" $i)
	otool -L $FRAMEWORK_FILE | grep -o /usr/.*Qt.*framework/\\w* | while read -a libs ; do
	       	install_name_tool -change ${libs[0]} @loader_path/../../../`basename ${libs[0]}`.framework/`basename ${libs[0]}` $FRAMEWORK_FILE
	done
done

declare -a BROKEN_PLUGINS;
k=0;
BUILD_PLUGINS_FOLDER_PATH="$BUILD_FOLDER_PATH/PlugIns"
for j in $(find ${BUILD_PLUGINS_FOLDER_PATH} -name *.dylib); do
	BROKEN_PLUGINS[${k}]=$j
	((k=k+1))
done
for i in "${BROKEN_PLUGINS[@]}"; do
	FRAMEWORK_FILE=$i
	otool -L $FRAMEWORK_FILE | grep -o /usr/.*Qt.*framework/\\w* | while read -a libs ; do
	       	install_name_tool -change ${libs[0]} @loader_path/../../Frameworks/`basename ${libs[0]}`.framework/`basename ${libs[0]}` $FRAMEWORK_FILE
	done
done

