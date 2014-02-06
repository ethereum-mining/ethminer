#!/bin/bash

set -e
rm -f ../cpp-ethereum_*_source.changes
debuild -S -sa
cd ..
dput -f ppa:ethereum/ethereum cpp-ethereum_*_source.changes

