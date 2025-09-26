#!/usr/bin/env bash

source ./util/util-functions.sh

# this script counts annotations from the Index Checker in the case studies from the ISSTA 2018 paper

# dependencies: git, make, mvn, grep, python

clone_all

# plume-lib

cd plume-lib/java
git checkout annotation-stats -q
make typecheck-only -s &> plume-stats-out
echo "plume-lib annotations:"
cat plume-stats-out | grep -e "index" -e "value" | grep -v -e ":annotation" -e "Statically" | grep -v -e "make" > plume-stats
python ../../util/read-anno-stats.py $PLUMELOC plume-stats
cd ../..
echo ""

# jfreechart

cd jfreechart
mvn clean -B -q
echo "jfreechart annotations:"
mvn compile -Dcheckerframework.checkers=org.checkerframework.common.util.count.AnnotationStatistics '-Dcheckerframework.extraargs=-Anolocations' '-Dcheckerframework.extraargs2=-Aannotations'  -B -q &> jfc-stats-out
cat jfc-stats-out | grep -e "index" -e "value" | grep -v -e ":annotation" -e "Statically" > jfc-stats
python ../util/read-anno-stats.py $JFCLOC jfc-stats
cd ..
echo ""

# guava

cd guava/guava
mvn clean -B -q
echo "guava annotations:"
mvn compile -P checkerframework-local -Dcheckerframework.checkers=org.checkerframework.common.util.count.AnnotationStatistics '-Daddtionalargs=-AonlyDefs=^com\.google\.common\.(?:base|primitives)\.' '-Dcheckerframework.extraargs=-Anolocations' '-Dcheckerframework.extraargs2=-Aannotations' -B -q &> guava-stats-out
cat guava-stats-out | grep -e "index" -e "value" | grep -v -e ":annotation" -e "Statically" > guava-stats
python ../../util/read-anno-stats.py $GUAVALOC guava-stats
cd ../..
echo ""

# totals

rm -f all-stats || true
cat guava/guava/guava-stats >> all-stats
cat jfreechart/jfc-stats >> all-stats
cat plume-lib/java/plume-stats >> all-stats

echo "annotation and density totals"
echo ""

python util/read-anno-stats.py `echo "$GUAVALOC + $JFCLOC + $PLUMELOC" | bc` all-stats 
