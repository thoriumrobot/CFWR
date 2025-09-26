#!/usr/bin/env bash

source ./util/util-functions.sh

# this script executes the commands to run the Index Checker on the three case studies from the ISSTA 2018 paper. Expected run-time is about 20 minutes.

# dependencies: git, make, mvn

# get all repos if not present
clone_all

# plume-lib

cd plume-lib/java
git checkout index-only
make typecheck-only
cd ../../

# guava

cd guava
mvn compile
cd ..

# jfreechart

cd jfreechart
mvn compile
cd ..
