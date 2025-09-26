# Index Checker reproduction package

This is the reproduction package for the paper Lightweight Verification of Array Indexing, appearing at ISSTA 2018. A copy of our current draft of the paper, with numbers updated from our original submission based on the progress we've made since then, is in this reproduction package as issta2018draft.pdf.

The reproduction package contains scripts and instructions to reproduce the numbers appearing in the paper, that are obtained by running the Index Checker or derived from code annotated and typechecked by the Index Checker. It also contains pointers to the open-source implementation of the tool and to the bugs that the Index Checker found in open-source projects.

## Obtaining the Implementation

The Index Checker itself is distributed with the Checker Framework at https://checkerframework.org/ .
The Index Checker's manual is at https://checkerframework.org/manual/#index-checker .

The Index Checker's source code is at
https://github.com/typetools/checker-framework/tree/master/checker/src/main/java/org/checkerframework/checker/index .
Its regression tests are at https://github.com/typetools/checker-framework/tree/master/checker/tests/index .

## Reproducing the results

Prerequisites:
Install bash, git, hg (Mercurial), make, mvn (Maven), python (scripts are written in python 2), and JDK 8. GNU coreutils is also required for MacOS.

The scripts should work on any Unix system, but we’ve only tested them on MacOS, Ubuntu, and Debian.

To obtain most of the numbers in Tables 3 and 4, run the script `reproduce-all.sh`.  This script may take about 30 minutes to run.

### Typechecking each case study

The `typecheck-all.sh script clones and type checks each case study using the Index Checker, and should issue no errors or warnings.  reproduce-all.sh does not run this script directly. However, it is useful for e.g. timing how long the case studies take to typecheck.

### Table 3

Table 3 contains a summary of the case studies in section 4. The reproduce-all.sh script will produce most of the numbers. The rest of this section discusses the others.

Lines of code: we counted non-comment, non-blank lines of Java code using cloc (http://cloc.sourceforge.net/), on clean (i.e. unannotated) copies of each case study repository before we started annotating.

Bugs fixed: Each of the following pull requests fixes one or more bugs in one of the case study programs, and was accepted by the maintainers of that program:

* https://github.com/google/guava/pull/3027
* https://github.com/jfree/jfreechart/pull/70
* https://github.com/jfree/jfreechart/pull/76

Plume-lib is maintained by one of the authors of this paper, so the bug fixes for it are commits. Each of the following commits contains one or more bug fixes:

* https://github.com/mernst/plume-lib/commit/0abfea178a813c3895fecebcbb8bcbed1cee1b42
* https://github.com/mernst/plume-lib/commit/f28b4a0ac4a7c683b7a7ecaeb9f6e734c5856c86
* https://github.com/mernst/plume-lib/commit/85530265d70ec7d3462c5a95e29eaff9c979194e
* https://github.com/mernst/plume-lib/commit/b5df093053146bcf32f485eea974e70cc854b407
* https://github.com/mernst/plume-lib/commit/16b8501d05e2458a95be3d164b8f708c4dc53183
* https://github.com/mernst/plume-lib/commit/7a6f75bf5a8b72a6ae823d4bd29a43742eb5cf50
* https://github.com/mernst/plume-lib/commit/d5355621ed0bd8872db2530e07d6a06b6518688f
* https://github.com/mernst/plume-lib/commit/e813d29
* https://github.com/mernst/plume-lib/commit/8224707
* https://github.com/mernst/plume-lib/commit/13b8045
* https://github.com/mernst/plume-lib/commit/95b3cab
* https://github.com/mernst/plume-lib/commit/339602f
* https://github.com/mernst/plume-lib/commit/4f10607

Bugs detected: This is the sum of the bugs that we fixed and the unreported bugs we found in JFreeChart. JFreeChart's maintainers rarely address pull requests. To avoid spamming, we try to avoid submitting more than one PR at a time. As PRs are closed, we report additional bugs. We have a pull request open that JFreeChart has not yet addressed: https://github.com/jfree/jfreechart/pull/78

All of the other bugs in JFreeChart have a comment next to the suppressed warning with the word "bug" in it, so grepping for them is straightforward (though there is a bit of noise). We counted the results of the following command manually (and manually examined each bug to confirm it was a true positive):

> cd jfreechart
> grep -Ri "bug" src/

False positives: This is the number of suppressed warnings in the case studies, less the number of suppressed true positives. reproduce-all.sh produces the number of suppressed warnings, so for JFreeChart this number is the result of that script minus 53.

False positive %: = false positives / non-trivial checks

### Table 4

reproduce-all.sh produces these tables as an intermediate step.

### False positive categories

The script print-fps.sh approximates the set of all false positives and groups them by the comment that describes their cause, sorting first by project and then by how common the false positives are. Because it needs information in comments, it uses grep instead of an accurate counter - so some other lines will also be included in the output. The false positives counts in section 4 can be obtained from this script's output.

### Comparing to other tools (section 5)

We compared to three other tools:

* FindBugs (http://findbugs.sourceforge.net/) version 3.0.1. It's easy to run FindBugs on our case studies. We provide a copy of the version of FindBugs we used. Run the script find-bugs.sh to run FindBugs on each of the case studies. The FindBugs configuration we used to only look for bounds errors is rangeonly.xml.

* KeY (https://www.key-project.org/) version 2.6.3, backed by Z3 version 4.5.1. We used KeY's provided GUI. The subject programs we used are found in the /minimized-bugs/buggy-key and /minimized-bugs/fixed-key directories (the first contains the buggy minimized test cases, and the latter contains the fixed versions - see section 5.2 of the paper). We manually attempted to run each proof to completion in KeY's default configuration. When a proof failed, we recorded the number of goals that KeY was not able to prove, and compared that number for the buggy and fixed versions. If the numbers were different, we counted that as a true positive for KeY.

* Clousot (https://www.microsoft.com/en-us/research/project/code-contracts/), Code Contracts version 1.9.10714.2 extension to Microsoft Visual Studio Enterprise 2015 838 Version 14.0.25431.01 Update 3 with Microsoft .NET Framework 839 Version 4.7.02556. We used Clousot via the Visual Studio plugin, and manually tested it on each of the translated benchmarks. The directory /minimized-bugs/buggy-clousot contains translated minimized bugs (section 5.2). The directory /test-suites/index-csharp contains the Index Checker's regression tests translated into C# (section 5.3). To run the Java versions of these tests on the Index Checker, run the script index-tests.sh. The subset of Clousot’s test suite that concerns array bounds in C# is in /test-suites/clousot-csharp. We translated this to Java (section 5.4); the translated tests are in /test-suites/clousot-java. To run the Index Checker on this test suite, run clousot-tests-count.sh.

### Description of each script in this repository

* clousot-tests-count.sh: this script executes the commands to run the Index Checker on a part of Clousot test suite translated to Java and count the number of expected and unexpected errors

* count-annotatable-locs.sh: this script uses a Checker Framework visitor to count how many possible annotatable locations there are in each case study

* count-annotations.sh: this script uses a Checker Framework visitor to record how many of each Index Checker annotation is in each case study. It also computes the "density" of each annotation by dividing the annotation count by the number of lines of code.

* count-java-casts.sh: this script uses a Checker Framework visitor to count cast AST nodes in each case study

* count-non-trivial-checks.sh: this script typechecks each case study and records how many of the checks made by the Index Checker are "non-trivial" - that is, involve a type other than the top or bottom type of the type system.

* create-zip.sh: builds the issta2018artifacts.zip file

* find-bugs.sh: runs FindBugs on each case study and prints the results

* fp-count.sh: uses a Checker Framework visitor to find suppressed warnings in each case study. Since 53 true positives are suppressed in JFreeChart, the false positive count for that case study is actually 53 less than the output of this script.

* index-tests.sh: this runs the Index Checker test suite

* print-fps.sh: greps for all suppressed warnings in all case studies and aggregates the results by cause. The result is messy, because this uses regular expressions instead of a Checker Framework visitor. Can be manually inspected to determine common false positives.

* reproduce-all.sh: produces results in tables 3 and 4 by delegating to other scripts.

* typecheck-all.sh: runs the Index Checker on each case study
