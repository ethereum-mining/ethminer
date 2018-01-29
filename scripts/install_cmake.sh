#!/usr/bin/env sh

# This script downloads the CMake binary and installs it in $PREFIX directory
# (the cmake executable will be in $PREFIX/bin). By default $PREFIX is
# ~/.local but can we changes with --prefix <PREFIX> argument.

# This is mostly suitable for CIs, not end users.

set -e

VERSION=3.8.1

if [ "$1" = "--prefix" ]; then
    PREFIX="$2"
else
    PREFIX=~/.local
fi

OS=$(uname -s)
case $OS in
Linux)  SHA256=10ca0e25b7159a03da0c1ec627e686562dc2a40aad5985fd2088eb684b08e491;;
Darwin) SHA256=cf8cf16eb1281127006507e69bbcfabec2ccbbc3dfb95888c39d578d037569f1;;
esac


BIN=$PREFIX/bin

if test -f $BIN/cmake && ($BIN/cmake --version | grep -q "$VERSION"); then
    echo "CMake $VERSION already installed in $BIN"
else
    FILE=cmake-$VERSION-$OS-x86_64.tar.gz
    VERSION_SERIES=$(echo $VERSION | awk '{ string=substr($0, 1, 3); print string; }')
    URL=https://cmake.org/files/v$VERSION_SERIES/$FILE
    ERROR=0
    TMPFILE=$(mktemp --tmpdir cmake-$VERSION-$OS-x86_64.XXXXXXXX.tar.gz)
    echo "Downloading CMake ($URL)..."
    curl -s "$URL" > "$TMPFILE"

    if type -p sha256sum > /dev/null; then
        SHASUM_TOOL="sha256sum"
    else
        SHASUM_TOOL="shasum -a256"
    fi

    SHASUM=$($SHASUM_TOOL "$TMPFILE")
    if ! (echo "$SHASUM" | grep -q "$SHA256"); then
        echo "Checksum mismatch!"
        echo "Actual:   $SHASUM"
        echo "Expected: $SHA256"
        exit 1
    fi
    mkdir -p "$PREFIX"
    echo "Unpacking CMake to $PREFIX..."
    tar xzf "$TMPFILE" -C "$PREFIX" --strip 1
    rm $TMPFILE
fi
