#!/usr/bin/env bash

source ./util/util-functions.sh

# this script runs FindBugs version 3.0.1 on each case study. It is expected to produce no output, since FindBugs neither has any false positives nor any true positives related to arrays over these case studies.

FINDBUGS=`pwd`/findbugs-3.0.1/lib/findbugs.jar
RANGEONLYXML=`pwd`/rangeonly.xml
SERVLET=`pwd`/javax.servlet-api-3.1.0.jar

if [ ! -d plume-lib ]; then
    echo "you must run reproduce-all.sh first so that the case studies are compiled before running findbugs"
    exit 1
fi

# plume-lib

cd plume-lib/java/src
java -jar $FINDBUGS -textui -include $RANGEONLYXML -auxclasspath ../lib plume
cd ../../..

# jfreechart

cd jfreechart
# mvn compile -B -q
cd target/classes/org/jfree
java -jar $FINDBUGS -textui -include $RANGEONLYXML -auxclasspath $SERVLET chart data
cd ../../../../../

# guava
# findbugs doesn't run on the annotated version of Guava correctly on all platforms
rm -rf guava
echo "removing guava to run findbugs on the unannotated version from Google, since the annotated versions don't interact well with findbugs on some platforms"
git clone https://github.com/google/guava -q
cd guava/guava/
# mvn compile -B -q "-Dcheck.index.phase=none"
mvn compile -B -q
cd target/classes/com/google/common
java -jar $FINDBUGS -textui -include $RANGEONLYXML base primitives
cd ../../../../../../../
rm -rf guava
