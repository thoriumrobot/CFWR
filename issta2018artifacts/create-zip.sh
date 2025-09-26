#!/usr/bin/env bash

source ./util/util-functions.sh

# this script creates the submission to the artificat evaluation for ISSTA 2018

rm -f *~ || true
rm -rf guava jfreechart plume-lib jsr308 || true
rm -f util/*~ || true
rm -f all-stats || true
cd ..
zip -r issta2018artifacts.zip issta2018artifacts/
echo "zipping complete. Find the resulting zip one directory above the directory this script ran in."
