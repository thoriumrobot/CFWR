Currently, this project reads the warnings from Checker Framework and calls Specimin on the right field or method.

Usage:

Replace ${...} with paths.

project-root=...
warning-log-file=...
CFWR-root=...

./gradlew run -PappArgs="${project-root} ${warning-log-file} ${CFWR-root}"

OR

mvn clean compile exec:java -Dexec.args="${project-root} ${warning-log-file} ${CFWR-root}"

Example:

./gradlew run -PappArgs="/home/ubuntu/naenv/checker-framework/checker/tests/index/ /home/ubuntu/naenv/checker-framework/index1.out /home/ubuntu/naenv/CFWR/"

OR

mvn clean compile exec:java -Dexec.args="/home/ubuntu/naenv/index_sub/ /home/ubuntu/naenv/checker-framework/index1.out /home/ubuntu/naenv/CFWR/"
