#!/bin/bash

dist="saucy"
version=$1

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

if [ "$1" == "" ]; then
	archdir="cpp-ethereum-$(date +%Y%m%d)"
else
	archdir="cpp-ethereum-$version"
fi
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
