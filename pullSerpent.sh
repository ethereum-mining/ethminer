#!/bin/bash

opwd="$PWD"
cd ../serpent
git pull
cp bignum.* compiler.* funcs.* lllparser.* opcodes.h parser.* rewriter.* tokenize.* util.* ../cpp-ethereum/libserpent/
cp cmdline.* "$opwd/sc/"
cp pyserpent.* "$opwd/libpyserpent/"
cd "$opwd"
perl -i -p -e 's:include "(.*)":include <libserpent/$1>:gc' sc/* libpyserpent/*

