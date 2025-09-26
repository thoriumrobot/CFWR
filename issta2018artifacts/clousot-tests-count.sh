#!/usr/bin/env bash

source ./util/util-functions.sh

# this script executes the commands to run the Index Checker on a part of Clousot test suite translated to Java and count the number of expected and unexpected errors

# dependencies: python2

# get CF source if not present
clone_and_build_cf

cd test-suites
python2 count.py clousot-java

