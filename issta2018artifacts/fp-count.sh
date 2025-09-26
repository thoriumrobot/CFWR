#!/usr/bin/env bash

source ./util/util-functions.sh

# this script counts the number of warnings in all case studies, and then
# subtracts the number of known bugs (and does some other cleaning).
# It prints the false positive count to the terminal.

clone_all

cd plume-lib/java
git checkout jcs -q
echo "plume-lib:"
make typecheck-only | grep "suppression"
cd ../../

# jfreechart

cd jfreechart
mvn clean -B -q
echo "jfreechart (note that this includes true positives that have not been fixed upstream)"
mvn compile -Dcheckerframework.checkers=org.checkerframework.common.util.count.JavaCodeStatistics  -B -q | grep "suppression"
cd ..

# guava

cd guava/guava
mvn clean -B -q
echo "guava:"
mvn compile -P checkerframework-local -Dcheckerframework.checkers=org.checkerframework.common.util.count.JavaCodeStatistics '-Daddtionalargs=-AonlyDefs=^com\.google\.common\.(?:base|primitives)\.' -B -q | grep "suppression"
cd ../..
