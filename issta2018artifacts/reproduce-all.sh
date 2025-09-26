#!/usr/bin/env bash

source ./util/util-functions.sh

# this script runs all the scripts to reproduce the numbers for the ISSTA 2018 paper, roughly in the order they appear in the paper

echo "Tables 3 and 4:"

./count-annotatable-locs.sh
./count-annotations.sh
./fp-count.sh
./count-non-trivial-checks.sh
./count-java-casts.sh
