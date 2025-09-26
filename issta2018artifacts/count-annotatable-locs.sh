#!/usr/bin/env bash

source ./util/util-functions.sh

# this script counts the number of locations that could be annotated in the case studies for the ISSTA 2018 paper.

# dependencies: git

# get all repos if not present
clone_all

# plume-lib
cd plume-lib/java
git checkout anno-locs -q
make clean -s
echo "plume-lib annotatable locations:"
make typecheck-only -s | wc -l
cd ../..

# jfreechart

cd jfreechart
mvn clean -B -q
echo "jfreechart annotatable locations:"
mvn compile -Dcheckerframework.checkers=org.checkerframework.common.util.count.AnnotationStatistics -B -q | wc -l
cd ..

# guava

cd guava/guava
mvn clean -B -q
echo "guava annotatable locations:"
mvn compile -P checkerframework-local -Dcheckerframework.checkers=org.checkerframework.common.util.count.AnnotationStatistics '-Daddtionalargs=-AonlyDefs=^com\.google\.common\.(?:base|primitives)\.' -Dcheck.index.phase=none -B -q | wc -l
cd ../..
