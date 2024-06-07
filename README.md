Currently, this project reads the warnings from Checker Framework and calls Specimin on the right Field or Method.

Usage:

mvn clean compile exec:java -Dexec.args="<project root> <warning log file> <CFWR root>"

Example:

mvn clean compile exec:java -Dexec.args="/home/ubuntu/naenv/index_sub/ /home/ubuntu/naenv/checker-framework/index1.out /home/ubuntu/naenv/CFWR/"
