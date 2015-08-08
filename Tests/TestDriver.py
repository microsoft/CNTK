#!/usr/bin/env python
# ----------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# ---------------------------------------------------------
# This is a test driver for running end-to-end CNTK tests
#
# ----- Running a test and/or updating baselines ------
# For instructions see:
# ./TestDriver.py --help
#
# ---- Adding the tests: -------
# File system organization:
#   Each test suite (e.g. Speech) has its own directory inside Tests
#   Each test (e.g. QuickE2E) has its own directory within test suite
#
# Each test directory has a following components:
#    - testcases.yml - main test confuguration file, whcih defines all test cases
#    - run-test  - (run-test) script
#    - baseline*.txt - baseline files whith a captured expected output of run-test script
#
# ----- testcases.yml format -------
# dataDir: <path> #<relative-path-to the data directory
#
# testCases:
#   <name of the testcase 1>:  
#     patterns:
#       - <pattern 1> # see pattern language
#       - <pattern 2>
#       - .....
#
#   <name of the testcase 2>:
#     patterns:
#       - <pattern 1>
#       - <pattern 2>
#       - .....
#   .....
#
# ----- pattern language --------
# Multpile patterns of the same testcase are matching a *single* line of text
# Pattern is essentiually a substring which has to be found in a line
# if pattern starts with ^ then matching is constrained to look only at the beginning of the line
#
# pattern can have one or multiple placelohders wrapped with double-curly braces:  {{...}}
# this placeholders can match any text conforming to the type constraint. Available placeholders
#  {{integer}} - matches any (positive or negative integer) value
#  {{float}} - matches any float value
#  {{float,tolerance=0.00001}} - matches float value with given absolute tolerance: 0.00001 in this example
#  {{float,tolerance=2%}} - matches float value with relative tolerance, 2% in this example
#
# At runtime patterns are compiled by TestDriver.py to regular expressions
#
# ---- Baseline files ----
# Order of searching baseline files, depends on the current mode for a given test:
#
#   1. baseline.<flavor>.<device>.txt
#   2. baseline.<flavor>.txt
#   3. baseline.<device>.txt
#   4. baseline.txt
#        where <flavor> = { debug | release }
#              <device> = { cpu | gpu }
# 
# ----- Algorithm ------
# Baseline verification:
#   For each testcase 
#     - filter all lines which matches
#       - if no lines found then abord with an error - since either baseline and/or pattern are invalid
# Running test:
#    Run test script (run-test) and capture output:
#
#    For each testcase
#      - filter all matching lines from baseline 
#      - filter all matching lines from test output
#      - compare filtered lines one by one, ensuring that substrings defined by patterns are matching
#
# In practice, TestDriver performs 1 pass through the output of run-test performing a real-time 
# matching against all test-cases/pattern simulteneously
#

import sys, os, argparse, traceback, yaml, subprocess, random, re, time

thisDir = os.path.dirname(os.path.realpath(__file__))

# This class encapsulates an instance of the test
class Test:
  # "Suite/TestName" => instance of Test
  allTestsIndexedByFullName = {} 

  # suite - name of the test suite
  # name - name of the test
  # path to the testcases.yml file
  def __init__(self, suite, name, pathToYmlFile):
    self.suite = suite
    self.name = name
    self.fullName = suite + "/" + name
    # computing location of test directory (yml file directory)
    self.testDir = os.path.dirname(pathToYmlFile)
    # parsing yml file with testcases 
    with open(pathToYmlFile, "r") as f:
      self.rawYamlData = yaml.safe_load(f.read())
 
    # finding location of data directory
    if self.rawYamlData["dataDir"]:
      self.dataDir = os.path.realpath(os.path.join(self.testDir, self.rawYamlData["dataDir"]))
    else:
      self.dataDir = self.testDir

    testCasesYaml = self.rawYamlData["testCases"]
    self.testCases = []
    for name in testCasesYaml.keys():
      try:
        self.testCases.append(TestCase(name, testCasesYaml[name]))
      except Exception as e:
        print >>sys.stderr, "ERROR registering test case: " + name
        raise

  # Populates Tests.allTestsIndexedByFullName by scanning directory tree
  # and finding all testcases.yml files
  @staticmethod
  def discoverAllTests():
    for dirName, subdirList, fileList in os.walk(thisDir):
      if 'testcases.yml' in fileList:
        testDir = dirName
        testName = os.path.basename(dirName)
        suiteDir = os.path.dirname(dirName)
        # sute name will be derived from the path components
        suiteName = os.path.relpath(suiteDir, thisDir).replace('\\', '/')
        try:
          test = Test(suiteName,  testName, dirName + "/testcases.yml")
          Test.allTestsIndexedByFullName[test.fullName.lower()] = test
        except Exception as e:
          print >>sys.stderr, "ERROR registering test: " + dirName
          traceback.print_exc()
          sys.exit(1)

  # Runs this test
  #   flavor - "debug" or "release"
  #   device - "cpu" or "gpu"
  #   args - command line arguments from argparse
  # returns an instance of TestRunResult
  def run(self, flavor, device, args):
    # Locating and reading baseline file
    baselineFile = self.findBaselineFile(flavor, device)
    if baselineFile == None:
      return TestRunResult.fatalError("Baseline file sanity check", "Can't find baseline file")

    with open(baselineFile, "r") as f:
      baseline = f.read().split("\n")
      if args.verbose:
         print "Baseline:", baselineFile

    # Before running the test, pre-creating TestCaseRunResult object for each test case
    # and compute filtered lines from baseline file.
    # Note: some test cases might fail at this time if baseline and/or patterns are inconsistant
    result = TestRunResult()
    result.succeeded = True
    if not args.update_baseline:
      for testCase in self.testCases:
        testCaseRunResult = testCase.processBaseline(baseline)
        if not testCaseRunResult.succeeded:
           result.succeeded = False
        result.testCaseRunResults.append(testCaseRunResult)
  
    # preparing run directory
    runDir = os.path.join(args.run_dir, "{0}_{1}@{2}_{3}".format(self.suite, self.name, flavor, device))
    if not os.path.isdir(runDir):
      os.makedirs(runDir)

    # preparing environment for the test script
    os.environ["TEST_FLAVOR"] = flavor
    os.environ["TEST_DEVICE"] = device
    os.environ["TEST_BUILD_LOCATION"] = args.build_location
    os.environ["TEST_DIR"] = self.testDir
    os.environ["TEST_DATA_DIR"] = self.dataDir
    os.environ["TEST_RUN_DIR"] = runDir
    # WORKAROUND: changing current dir to the dataDir so relative paths in SCP files work as expected
    os.chdir(self.dataDir)
    # Running test script
    #TODO:port this properly to windows
    # Writing standard output to the file and to the console (if --verbose)
    logFile = os.path.join(runDir, "output.txt")
    allLines = []
    if args.verbose:
      print self.fullName + ":>" + logFile
    with open(logFile, "w") as output:
      cmdLine = ["bash", "-c", self.testDir + "/run-test 2>&1"]
      process = subprocess.Popen(cmdLine, stdout=subprocess.PIPE)

      while True:
        line = process.stdout.readline() 
        if not line:
           break

        if len(line)>0 and line[-1]=='\n':
          line=line[:len(line)-1]

        if args.verbose:
          print self.fullName + ": " + line

        print >>output, line
        allLines.append(line)
        output.flush()
        for testCaseRunResult in result.testCaseRunResults:
          testCaseRunResult.testCase.processLine(line, testCaseRunResult, args.verbose)

    exitCode = process.wait()
    success = True

    # checking exit code
    if exitCode != 0:
      return TestRunResult.fatalError("Exit code must be 0", "==> got exit code {0} when running: {1}".format(exitCode, " ".join(cmdLine)), logFile = logFile)

    # saving log file path, so it can be reported later
    result.logFile = logFile

    # finalizing verification - need to check whether we have any unmatched lines
    for testCaseRunResult in result.testCaseRunResults:
      testCaseRunResult.testCase.finalize(testCaseRunResult)
      if not testCaseRunResult.succeeded:
        result.succeeded = False

    if args.update_baseline and result.succeeded:
      # When running in --update-baseline mode 
      # verifying that new output is succesfully matching every pattern in the testcases.yml
      # If this is not the case then baseline update will be rejected
      for testCase in self.testCases:
        testCaseRunResult = testCase.processBaseline(allLines)
        if not testCaseRunResult.succeeded:
           result.succeeded = False
        result.testCaseRunResults.append(testCaseRunResult)

      if result.succeeded:
       if args.verbose:
         print "Updating baseline file", baselineFile
       with open(baselineFile, "w") as f:
         f.write("\n".join(allLines))

    return result

  # Finds a location of a baseline file by probing different names in the following order:
  #   baseline.$flavor.$device.txt
  #   baseline.$flavor.txt
  #   baseline.$device.txt
  #   baseline.txt
  def findBaselineFile(self, flavor, device):
    for f in ["." + flavor.lower(), ""]:
      for d in ["." + device.lower(), ""]:
        candidateName = "baseline" + f + d + ".txt";
        fullPath = os.path.join(self.testDir, candidateName)
        if os.path.isfile(fullPath):
           return fullPath
    return None

# This class encapsulates one testcase (in testcases.yml file)
class TestCase:
  def __init__(self, name, yamlNode):
    self.name = name
    self.patterns = []
    if "patterns" in yamlNode:
      for pattern in yamlNode["patterns"]:
        try:
          self.patterns.append(TestPattern(pattern))
        except Exception as e:
          print >>sys.stderr, "ERROR registering pattern: " + pattern
          raise

  # Processes the baseline file and return an instance of TestCaseRunResult
  # which is ready to be passed into processLine
  def processBaseline(self, baseline):
    result = TestCaseRunResult(self.name, True)
    result.diagnostics = ""
    result.testCase = self
 
    # filter all lines of baseline file leaving only those which match ALL the patterns   
    filteredLines = []
    for line in baseline:
      if all([p.match(line) for p in self.patterns]):
        filteredLines.append(line)
    if len(filteredLines) == 0:
       result.succeeded = False
       result.diagnostics+="Baseline file doesn't have any lines matching all patterns defined in the test case.\n"\
                           "Possible cause: patterns are wrong and/or baseline file doesn't have required line"
    result.expectedLines = filteredLines
    return result
     
  # Runs this test case and report result into TestCaseRunResult
  def processLine(self, line, result, verbose):
    if all([p.match(line) for p in self.patterns]):
      if len(result.expectedLines) > 0:
        # we have mathed line in the output and at leat one remaining unmatched in a baseline
        expected = result.expectedLines[0]
        # running comparison logic for each pattern
        failedPatterns = []
        for p in self.patterns:
          if not p.compare(expected, line):
            result.succeeded = False
            failedPatterns.append(p)

        # in the case of failure - reporting mismatched lines
        if len(failedPatterns)>0:
          result.diagnostics+=("Baseline: {0}\n"+
                               "Output:   {1}\n"
                              ).format(expected, line)
          if verbose:
            print "[FAILED]: Testcase", self.name
            print "Baseline:", expected
     
          # also show all failed patterns
          for p in failedPatterns:
            msg = "Failed pattern: " + p.patternText
            if verbose:
              print msg
            result.diagnostics+=msg+"\n"
        # removing this line, since we already matched it (whether succesfully or not - doesn't matter)
        del result.expectedLines[0]
      else:
        # we have matched line in the output - but don't have any remaining unmatched in a baseline
        result.succeeded = False
        result.diagnostics+=("Unexpected (extra) line in the output which matches the pattern, but doesn't appear in baseline file.\n"+
                             "Extra line: {0}"
                            ).format(line)

  # called once for each TestCaseRunResult at the end to check for unmatched patterns
  def finalize(self, result):
    if len(result.expectedLines) > 0:
       result.succeeded = False
       result.diagnostics+=("{0} expected lines weren't observed in the output.\n"+
                            "First unmatched: {1}"
                           ).format(len(result.expectedLines), result.expectedLines[0])

# This encapsulates parsing and evaluation of a test patterns occurring in testcases.yml file
class TestPattern:
  # maps a type (specified in {{...}} expressions) to a regular expression
  typeTable = {
     "integer" : r"\s*-?[0-9]+",
     "float"   : r"\s*-?([0-9]*\.[0-9]+|[0-9]+)(e[+-]?[0-9]+)?"
   }
  def __init__(self, patternText):
    self.patternText = str(patternText)
    if len(patternText) == 0: 
      raise Exception("Empty pattern")
    if patternText[0]=='^':
      patternText = patternText[1:]
      prefix = "^"
    else:
      prefix = ".*?"

    # After parsing this will be a list of tuples (dataType, tolerance) for each {{...}} section from left to right
    self.groupInfo = []

    # Transforming our pattern into a sigle regular expression
    # processing {{...}} fragments and escaping all regex special characters
    self.regexText = prefix + re.sub(r"(\{\{[^}]+\}\}|[\[\]\.\*\+\{\}\(\)\$\^\\\|\?])", self.patternParse, patternText)
    # Compiling it to perform a check (fail-fast) and for faster matching later
    self.regex = re.compile(self.regexText)


  # this is a callback method passed to re.sub call above - it performs the core parsing logic
  def patternParse(self, match):
    fragment = match.group(1)
    if len(fragment) == 1:
      # this is a spexcial character of regex
      return "\\" + fragment;
    else:
      # parsing {{...}} expressions
      m = re.match(r"{{(integer|float)(,tolerance=([-0-9\.e]*)(%?))?}}", fragment)
      dataType = m.group(1)
      if m.group(3):
        tolerance = float(m.group(3))
        if m.group(4) == "%":
          # using minus sign to indicate that it is a relative value
          tolerance = - tolerance/100.0;
      else:
        tolerance = 0.0
      # saving information about data type and tolerance
      self.groupInfo.append((dataType, tolerance))
      # converting this to regex which mathes specific type
      # All {{...}} sections are converted to regex groups named as G0, G1, G2...
      return "(?P<G{0}>{1})".format(len(self.groupInfo)-1, TestPattern.typeTable[dataType])
 
  # Checks wether given line matches this pattern
  # returns True or False
  def match(self, line):
    return self.regex.match(line) != None

  # Compares a line from baseline log and a line from real output against this pattern
  # return true or false
  def compare(self, expected, actual):
    em = self.regex.match(expected)
    am = self.regex.match(actual)
    if em == None and am == None:
       return True
    if em == None or am == None:
       return False

    for i in range(0, len(self.groupInfo)):
      dataType, tolerance = self.groupInfo[i]
      groupId = "G"+str(i)
      expectedText = em.group(groupId).strip()
      actualText = am.group(groupId).strip()
      if dataType=="integer":
        return int(expectedText) == int(actualText)
      elif dataType=="float":
        epsilon = tolerance if tolerance > 0 else abs(float(expectedText)*tolerance)
        return abs(float(expectedText)-float(actualText)) <= epsilon
      else:
        return False;
    return True

class TestRunResult:
  def __init__(self):
    self.succeeded = False;
    self.testCaseRunResults = [] # list of TestCaseRunResult
  
  @staticmethod
  def fatalError(name, diagnostics, logFile = None):
    r = TestRunResult()
    r.testCaseRunResults.append(TestCaseRunResult(name, False, diagnostics))
    r.logFile = logFile
    return r

class TestCaseRunResult:
  def __init__(self, testCaseName, succeeded, diagnostics = None):
    self.testCaseName = testCaseName
    self.succeeded = succeeded
    self.diagnostics = diagnostics
    self.expectedLines = [] # list of remaining unmatched expected lines from the baseline file for this test case run

# Lists all available tests
def listCommand(args):
  for t in Test.allTestsIndexedByFullName.values():
    print t.fullName

# Runs given test(s) or all tests
def runCommand(args):
  if len(args.test) > 0:
     testsToRun = []
     for name in args.test:
       if name.lower() in Test.allTestsIndexedByFullName:
         testsToRun.append(Test.allTestsIndexedByFullName[name.lower()])
       else:
         print >>sys.stderr, "ERROR: test not found", name
         return 1
  else:
     testsToRun = Test.allTestsIndexedByFullName.values()
  devices = ["cpu", "gpu"]
  if (args.device):
    args.device = args.device.lower()
    if not args.device in devices:
      print >>sys.stderr, "--device must be one of", devices
      return 1
    devices = [args.device]

  flavors = ["debug", "release"]
  if (args.flavor):
    args.flavor = args.flavor.lower()
    if not args.flavor in flavors:
      print >>sys.stderr, "--flavor must be one of", flavors
      return 1
    flavors = [args.flavor]

  print "CNTK Test Driver is started"
  print "Running tests:  ", " ".join([y.fullName for y in testsToRun])
  print "Build location: ", args.build_location
  print "Run location:   ", args.run_dir
  print "Flavors:        ", " ".join(flavors)
  print "Devices:        ", " ".join(devices)
  if (args.update_baseline):
    print "*** Running in automatic baseline update mode ***"
  print ""
  succeededCount, totalCount = 0, 0
  for test in testsToRun:
    for flavor in flavors:
      for device in devices:
        totalCount = totalCount + 1
        # Printing the test which is about to run (without terminating the line)
        sys.stdout.write("Running test {0} ({1} {2}) - ".format(test.fullName, flavor, device));
        # in verbose mode, terminate the line, since there will be a lot of output
        if args.verbose:
          sys.stdout.write("\n");
        sys.stdout.flush()
        # Running the test and collecting a run results
        result = test.run(flavor, device, args)
        if args.verbose:
          # writing the test name one more time (after possibly long verbose output)
          sys.stdout.write("Test finished {0} ({1} {2}) - ".format(test.fullName, flavor, device));
        if result.succeeded:
          succeededCount = succeededCount + 1
          # in no-verbose mode this will be printed in the same line as 'Running test...'
          print "[OK]"
        else:
          print "[FAILED]"
        # Showing per-test-case results:
        for testCaseRunResult in result.testCaseRunResults:
           if testCaseRunResult.succeeded:
             # Printing 'OK' test cases only in verbose mode
             if (args.verbose):
               print(" [OK] " + testCaseRunResult.testCaseName);
           else:
             # 'FAILED' + detailed diagnostics with proper indendtation
             print(" [FAILED] " + testCaseRunResult.testCaseName);
             if testCaseRunResult.diagnostics:
               for line in testCaseRunResult.diagnostics.split('\n'):
                 print "    " + line;
             # In non-verbose mode log wasn't piped to the stdout, showing log file path for conveniencce
               
        if not result.succeeded and not args.verbose and result.logFile:
          print "  See log file for details:", result.logFile
        
  if args.update_baseline:
    print "{0}/{1} baselines updated, {2} failed".format(succeededCount, totalCount, totalCount - succeededCount)
  else:
    print "{0}/{1} tests passed, {2} failed".format(succeededCount, totalCount, totalCount - succeededCount)
  if succeededCount != totalCount:
    sys.exit(10)

# ======================= Entry point =======================
parser = argparse.ArgumentParser(description="TestDriver - CNTK Test Driver")
subparsers = parser.add_subparsers(help="command to execute. Run TestDriver.py <command> --help for command-specific help")
runSubparser = subparsers.add_parser("run", help="run test(s)")
runSubparser.add_argument("test", nargs="*",
                    help="optional test name(s) to run, specified as Suite/TestName. "
                         "Use list command to list available tests. "
                         "If not specified then all tests will be run.")
#TODO: port paths to Windows
defaultBuildLocation=os.path.realpath(os.path.join(thisDir, "..", "build"))
runSubparser.add_argument("-b", "--build-location", default=defaultBuildLocation, help="location of the CNTK build to run")
runSubparser.add_argument("-d", "--device", help="cpu|gpu - run on a specific device")
runSubparser.add_argument("-f", "--flavor", help="release|debug - run only a specific flavor")
#TODO: port paths to Windows
defaultRunDir=os.path.join("/tmp", "cntk-test-{0}.{1}".format(time.strftime("%Y%m%d%H%M%S"), random.randint(0,1000000)))
runSubparser.add_argument("-r", "--run-dir", default=defaultRunDir, help="directory where to store test output, default: a random dir within /tmp")
runSubparser.add_argument("--update-baseline", action='store_true', help="update baseline file(s) instead of matching them")
runSubparser.add_argument("-v", "--verbose", action='store_true', help="verbose output - dump all output of test script")

runSubparser.set_defaults(func=runCommand)

listSubparser = subparsers.add_parser("list", help="list available tests")
listSubparser.set_defaults(func=listCommand)

if len(sys.argv)==1:
    parser.print_help()
    sys.exit(1)

args = parser.parse_args(sys.argv[1:])

# discover all the tests
Test.discoverAllTests()

# execute the command
args.func(args)

