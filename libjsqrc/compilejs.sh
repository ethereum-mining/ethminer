#!/bin/bash

cd ethereumjs
export PATH=$PATH:$1:$2
npm install
npm run-script build

