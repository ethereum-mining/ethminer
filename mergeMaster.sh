#!/bin/bash

git checkout master && git merge --no-ff $1+ && git push && git tag -f $1 && git push --tags -f

