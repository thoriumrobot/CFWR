#!/usr/bin/env bash

source ./util/util-functions.sh

# this script counts the number of non-trivial checks made by the Index Checker in the case studies for the ISSTA 2018 paper. A non-trivial check is one that involves a type that is neither bottom nor top in its type system (in other words, this counts how many "places in the source code" the Index Checker could issue a false positive, in the worst case. Note that this script takes some time to run - it is running the Index Checker and counting the checks as they occur.

# dependencies: git, grep, wc, make

# get all repos if not present
clone_all

# plume-lib
cd plume-lib/java
git checkout indexShowChecks -q
echo "plume-lib non-trivial checks:"
make typecheck-only | grep "expected" | grep -v "success" | grep -v "About to test" | grep -v "Unknown" | grep -v "FAIL" | wc -l
cd ../..

# jfreechart

cd jfreechart
mvn clean -B -q
echo "jfreechart non-trivial checks:"
mvn compile -Dcheckerframework.extraargs=-Ashowchecks -B -q | grep "expected" | grep -v "success" | grep -v "About to test" | grep -v "Unknown" | grep -v "FAIL" | wc -l
cd ..

# guava

cd guava/guava
mvn clean -B -q
echo "guava non-trivial checks:"
mvn compile -P checkerframework-local -Dcheckerframework.checkers=org.checkerframework.checker.index.IndexChecker -Dindex.only.arg=-Ashowchecks -B -q | grep "expected" | grep -v "success" | grep -v "About to test" | grep -v "Unknown" | grep -v "FAIL" | wc -l
cd ../..
