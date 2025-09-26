#!/usr/bin/env bash

source ./util/util-functions.sh

# this script prints the false positives, by category, in the Index Checker case studies for the ISSTA 2018 paper

# dependencies: git, grep, sed

clone_all

sh ./util/count-suppressions.sh index
