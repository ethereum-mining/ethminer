#!/bin/bash

if [[ ! "$3" == "" ]]; then
	debchange -v $1-$2 "$3"
	git commit -a -m "$3"
fi

opwd=`pwd`
cd /tmp

echo Checking out...
git clone $opwd
cd cpp-ethereum

if [ "$1" == "" ]; then
	archdir="cpp-ethereum-$(date +%Y%m%d)"
else
	archdir="cpp-ethereum-$1"
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

[[ ! "$1" == "" ]] && ln -sf $archfile "cpp-ethereum_$1.orig.tar.bz2"

echo Packaging...
cd "$archdir"
./package.sh

echo Cleaning up...
rm -rf /tmp/$archdir
mv /tmp/$archfile ~

echo Done.
