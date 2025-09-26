#!/usr/bin/env bash

source ./util/util-functions.sh

# this runs the Index Checker test suite

# get CF source if not present
clone_and_build_cf

cd $JSR308/checker-framework
./gradlew IndexTest
