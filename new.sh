#!/bin/bash

type="$1"
path="$2"
name="$3"

if ! [[ -n $type ]] || ! [[ -n $path ]] || ! [[ -n $name ]]; then
	echo "Usage new.sh <type> <path> <name>"
	echo "e.g. new.sh plugin alethzero MyPlugin"
	exit
fi

cd templates
for i in $type.*; do
	n="../$path/${i/$type/$name}"
	cp "$i" "$n"
	perl -i -p -e "s/\\\$NAME/$name/gc" "$n"
done



