#!/bin/bash

opwd="$PWD"
br=$(git branch | grep '\*' | sed 's/^..//')

n=cpp-ethereum-src-$(date "+%Y%m%d%H%M%S" --date="1970-01-01 $(git log -1 --date=short --pretty=format:%ct) sec GMT")-$(grep "Version = " libdevcore/Common.cpp | sed 's/^[^"]*"//' | sed 's/".*$//')-$(git rev-parse HEAD | cut -c1-6)

cd /tmp
git clone "$opwd" $n
cd $n
git checkout $br
rm -f package.sh
cd ..
tar c $n | bzip2 -- > $opwd/../${n}.tar.bz2
rm -rf $n
cd $opwd

echo "SHA1(${n}.tar.bz2) = $(shasum $opwd/../${n}.tar.bz2 | cut -d' '  -f 1)"

