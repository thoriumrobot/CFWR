#!/usr/bin/env bash

# utility functions shared by many bash scripts. To include this, use `source /absolute/path/to/this/file`.

set -o errexit
set -o pipefail
set -o nounset

# set all lines of code in one place
export PLUMELOC=14586
export JFCLOC=94233
export GUAVALOC=10694

function clone_guava () {
    if [ ! -d guava ]; then
	git clone "https://github.com/panacekcz/guava"
	cd guava
	git checkout hss
	cd ..
    else
	cd guava
	git pull -q
	cd ..
    fi
}

function clone_jfreechart () {
    if [ ! -d jfreechart ]; then
	git clone "https://github.com/kelloggm/jfreechart"
	cd jfreechart
	git checkout index
	cd ..
    else
	cd jfreechart
	git pull -q
	cd ..
    fi
}

function clone_plume () {
    if [ ! -d plume-lib ]; then
	git clone "https://github.com/kelloggm/plume-lib"
    else
	cd plume-lib
	git pull -q
	cd ..
    fi
}

function clone_and_build_cf () {
    if [ "x${CHECKERFRAMEWORK-}" == "x" ]; then
	if [ ! -d jsr308 ]; then
	    echo "No Checker Framework found"
	    export JSR308=`pwd`/jsr308
	    mkdir -p $JSR308
	    cd $JSR308
	    git clone "https://github.com/typetools/checker-framework"
	    export CHECKERFRAMEWORK=$JSR308/checker-framework
	    cd $CHECKERFRAMEWORK
	    echo "Building the Checker Framework"
	    ./gradlew cloneAndBuildDependencies
	    ./gradlew assemble
	    cd ../../
	else
	    export JSR308=`pwd`/jsr308
 	    export CHECKERFRAMEWORK=$JSR308/checker-framework
	fi
    fi	
}

function clone_all () {
    clone_plume
    clone_guava
    clone_jfreechart
    clone_and_build_cf
}
