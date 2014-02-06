#!/bin/bash

set -e
rm -f ../cpp-ethereum_*
debuild -S -sa
cd ..
dput -f ppa:r-launchpad-gavofyork-fastmail-fm/ethereum ethereum_*_source.changes

