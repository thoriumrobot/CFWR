#!/usr/bin/env bash

source ./util/util-functions.sh

# this script counts the number of java casts in the case studies for the ISSTA 2018 paper.

# dependencies: git

# get all repos if not present
clone_all

# plume-lib
cd plume-lib/java
git checkout jcs -q
echo "plume-lib:"
make typecheck-only | grep "cast"
cd ../..

# jfreechart

cd jfreechart
mvn clean -B -q
echo "jfreechart"
mvn compile -Dcheckerframework.checkers=org.checkerframework.common.util.count.JavaCodeStatistics  -B -q | grep "cast"
cd ..

# guava

cd guava/guava
mvn clean -B -q
echo "guava:"
mvn compile -P checkerframework-local -Dcheckerframework.checkers=org.checkerframework.common.util.count.JavaCodeStatistics '-Daddtionalargs=-AonlyDefs=^com\.google\.common\.(?:base|primitives)\.' -B -q | grep "cast"
cd ../..
