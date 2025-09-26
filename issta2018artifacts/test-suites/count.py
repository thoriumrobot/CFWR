import re
import subprocess
import os.path
import os
import sys

if len(sys.argv) != 2:
	print("USAGE: python count.py TESTDIR")
	exit()

jsr308 = os.environ["JSR308"]
src = sys.argv[1]

files = []
expected = []
found = []

# Search Java files for comments containing expected errors
for file in os.listdir(src):
	if file.endswith(".java"):
		path = os.path.join(src, file)
		files.append(path)
		with open(path) as javafile:
			lineno = 1
			for line in javafile:
				lineno+=1
				for match in re.finditer(' :: error: \\(([a-z.]+)\\)', line):
					expected.append((path, lineno, match.group(1)))

# Run the Index Checker and capture the output
checker_process = subprocess.Popen([os.path.join(jsr308,'checker-framework/checker/bin/javac'),'-Xmaxerrs', '10000', '-proc:only', '-processor', 'org.checkerframework.checker.index.IndexChecker'] + files, stderr=subprocess.PIPE)
checker_output,checker_error = checker_process.communicate()

# Search the output for errors
for line in checker_error.splitlines():
	match = re.match('^([^:]+):(\\d+): error:[^\\[]+\\[([a-z.]+)\\]', line)
	if match is not None:
		i = (match.group(1), int(match.group(2)), match.group(3))
		found.append(i)

# Check that expected errors were found and display the result
if not (set(expected) <= set(found)):
	print("FAILURE: expected errors not found")
	print("EXPECTED (%d)"%len(expected))
	for e in expected:
		print(e)
		if e not in found:
			print("NOT FOUND")
	print("FOUND (%d)"%len(found))
	for e in found:
		print(e)
else:
	print("SUCCESS %d expected %d false positives"%(len(expected), len(found)-len(expected)))
