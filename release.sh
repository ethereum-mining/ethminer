#!/bin/bash

dist="saucy"
version=$(grep "define ETH_VERSION" libethereum/Common.h | cut -d ' ' -f 3)
branch="$(git branch | grep \* | cut -c 3-)"

if [[ ! "$1" == "" ]]; then
	version=$1
fi

if [[ ! "$3" == "" ]]; then
	if [[ ! "$4" == "" ]]; then
		dist=$4
	fi
	if [[ "$2" == "-i" ]]; then
		# increment current debian release only
		# new version ./release VERSION -i MESSAGE DIST
		debchange -i -p "$3" -D "$dist"
		git commit -a -m "$3"
	else
		# new version ./release VERSION DEB-VERSION MESSAGE DIST
		debchange -v $version-$2 -p "$3" -D "$dist"
		git commit -a -m "$3"
	fi
fi

opwd=`pwd`
cd /tmp

echo Checking out...
git clone $opwd
cd cpp-ethereum
git checkout "$branch"

archdir="cpp-ethereum-$version"
archfile="$archdir.tar.bz2"

echo Cleaning backup files...
find . | grep \~ | xargs rm -f

echo Cleaning others...
rm release.sh

echo Cleaning versioning...
rm -rf .git .gitignore

echo Renaming directory...
cd ..
rm -rf $archdir
mv cpp-ethereum $archdir

echo Creating archive...
tar c $archdir | bzip2 -- > $archfile

[[ ! "$version" == "" ]] && ln -sf $archfile "cpp-ethereum_$version.orig.tar.bz2"

echo Packaging...
cd "$archdir"
./package.sh

echo Cleaning up...
rm -rf /tmp/$archdir
mv /tmp/$archfile ~

echo Done.
