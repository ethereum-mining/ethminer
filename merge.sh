#!/bin/bash

git checkout "$1+" && git merge --no-ff develop && git push && git tag -f $1 && git push --tags -f && git checkout develop

