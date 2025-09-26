#!/usr/bin/python2

# this script reads a list of annotation values and prints annotation densities

# usage: python2 read-anno-stats.py $LOC $ANNOTATIONS
# where LOC is the number of lines of code in the output
# and ANNOTATIONS is a file containing the output of the CF's AnnotationStatistics "checker"

from __future__ import division
import sys

if len(sys.argv) is not 3:
    print "usage: python read-anno-stats.py $LOC $ANNOTATIONS\nwhere LOC is the number of lines of code in the output\nand ANNOTATIONS is a file containing the output of the CF's AnnotationStatistics checker"
    exit(1)

loc = int(sys.argv[1])
fn = sys.argv[2]

total_count = 0

mpanno_count = {}

with open(fn) as fl:
    for ln in fl:
        ln = ln.strip()
        anno = ln.split()[0].strip().split('.')[-1]
        # padding
        if len(anno) < 8:
            anno = anno + "\t"
        if len(anno) < 16:
            anno = anno + "\t"
        count = int(ln.split()[1].strip())
        if anno in mpanno_count.keys():
            mpanno_count[anno] = str(int(mpanno_count[anno]) + count)
        else:
            mpanno_count[anno] = str(count)
        
        total_count = total_count + count

print "total annotations: " + str(total_count)
print "total density: 1/" + str(int(round(loc / total_count)))
print ""

print "annotation\t\tcount\tdensity"
print "----------\t\t-----\t-------"

for anno in mpanno_count.keys():
    idensity = int(round(loc / int(mpanno_count[anno])))
    print anno + "\t" + mpanno_count[anno] + "\t1/" + str(idensity)
