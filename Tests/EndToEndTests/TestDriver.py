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
#   Each test suite (e.g. Speech) has its own directory inside EndToEndTests
#   Each test (e.g. QuickE2E) has its own directory within test suite
#
# Each test directory has a following components:
#    - testcases.yml - main test configuration file, which defines all test cases
#    - run-test - (run-test) script
#    - baseline*.txt - baseline files with a captured expected output of run-test script
#
# ----- testcases.yml format -------
# isPythonTest: [True|False]
# dataDir: <path> #<relative-path-to the data directory
# tags: # optional tags - see tagging system
#   - <tag1> <optional-predicate>
#   - <tag2> <optional-predicate>
#   - ....
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
# Multiple patterns of the same testcase are matching a *single* line of text
# Pattern is essentially a substring which has to be found in a line
# if pattern starts with ^ then matching is constrained to look only at the beginning of the line
#
# pattern can have one or multiple placeholders wrapped with double-curly braces:  {{...}}
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
#   1. baseline.<os>.<flavor>.<device>.txt
#   2. baseline.<os>.<flavor>.txt
#   3. baseline.<os>.<device>.txt
#   4. baseline.<os>.txt
#   5. baseline.<flavor>.<device>.txt
#   6. baseline.<flavor>.txt
#   7. baseline.<device>.txt
#   8. baseline.txt
#        where <flavor> = { debug | release }
#              <device> = { cpu | gpu }
#
# Baseline files are optional. They only evaluate if test defines one or more pattern-driven test cases.
# If no test cases are defined, then TestDriver uses exit code of the run-test script as the only criteria
# of successful completion of the test.

# ----- Tagging system ------
# Unit tests can be optionally tagged with 1 or many tags
# CNTK build/test lab uses those tags to understand which tests to run during different flavors of build jobs (nightly, BVT, checkin)
#
# Tag can be optionally predicated with a python boolean expression over 'flavor' (debug/release), 'device' (cpu/gpu), 'os' (windows/linux) variables.
# this allows to restrict tagging of the test to specific combinations of those variables
#
# ----- Algorithm ------
# Baseline verification:
#   For each testcase
#     - filter all lines which matches
#       - if no lines found then abort with an error - since either baseline and/or pattern are invalid
# Running test:
#    Run test script (run-test) and capture output:
#
#    For each testcase
#      - filter all matching lines from baseline
#      - filter all matching lines from test output
#      - compare filtered lines one by one, ensuring that substrings defined by patterns are matching
#
# In practice, TestDriver performs 1 pass through the output of run-test performing a real-time
# matching against all test-cases/pattern simultaneously
#

from __future__ import print_function
import sys, os, argparse, traceback, yaml, subprocess, random, re, time, stat

thisDir = os.path.dirname(os.path.realpath(__file__))
windows = os.getenv("OS")=="Windows_NT"

def cygpath(path, relative=False):
    if windows:
        if path.startswith('/'):
          return path
        path = os.path.abspath(path)
        if not relative and path[1]==':': # Windows drive
          path = '/cygdrive/' + path[0] + path[2:]
        path = path.replace('\\','/')

    return path

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
    self.isPythonTest = False

    # computing location of test directory (yml file directory)
    self.testDir = cygpath(os.path.dirname(pathToYmlFile), relative=True)

    # parsing yml file with testcases
    with open(pathToYmlFile, "r") as f:
      self.rawYamlData = yaml.safe_load(f.read())

    # finding location of data directory
    if self.rawYamlData["dataDir"]:
      self.dataDir = os.path.realpath(os.path.join(self.testDir, self.rawYamlData["dataDir"]))
    else:
      self.dataDir = self.testDir

    # add Python test marker
    try:
      self.isPythonTest = bool(self.rawYamlData["isPythonTest"])
    except KeyError:
      pass

    # parsing test cases
    self.testCases = []
    if "testCases" in list(self.rawYamlData.keys()):
      testCasesYaml = self.rawYamlData["testCases"]
      for name in list(testCasesYaml.keys()):
        try:
          self.testCases.append(TestCase(name, testCasesYaml[name]))
        except Exception as e:
          print("ERROR registering test case: " + name, file=sys.stderr)
          raise

    # parsing all tags, example input:
    # tags:
    # - bvt-l  (flavor=='debug') ^ (device=='cpu')  # tag with a python predicate expression
    # - nightly-l  #tag without a predicate
    #
    # Predicate expressions must produce boolean value and may refer to following variables: flavor, device, os
    self.tags = {}
    if self.rawYamlData["tags"]:
      for tagLine in self.rawYamlData["tags"]:
        tagLineSplit = tagLine.split(' ', 1) # splitting tag name from predicate expression
        tagName = tagLineSplit[0].lower().strip()

        # using specified python expression (or 'True' if former isn't provided)
        pythonExpr = tagLineSplit[1] if len(tagLineSplit)==2 else "True"

        # converting python expression into lambda and doing a smoke test by calling it with dummy parameters
        predicate = lambda pythonExpr=pythonExpr, **kwargs: eval(pythonExpr, kwargs)
        try:
          assert(type(predicate(flavor='foo', device='bar', os='foobar', build_sku='qux')) == bool)
        except Exception as e:
          print("Can't parse tag predicate expression in {0} ({1}):\n{2}".format(pathToYmlFile, pythonExpr, e))
          raise e

        # saving generated lambda into tags dictionary
        self.tags[tagName] = predicate

  # Populates Tests.allTestsIndexedByFullName by scanning directory tree
  # and finding all testcases.yml files
  @staticmethod
  def discoverAllTests():
    for dirName, subdirList, fileList in os.walk(thisDir):
      # Temporarily disable these tests on Windows due to an issue introduced by adding onnx to our CI.
      if os.path.basename(dirName) == 'Keras' and windows:
        continue
      if 'testcases.yml' in fileList:
        testDir = dirName
        testName = os.path.basename(dirName)
        suiteDir = os.path.dirname(dirName)
        # suite name will be derived from the path components
        suiteName = os.path.relpath(suiteDir, thisDir).replace('\\', '/')
        try:
          test = Test(suiteName,  testName, dirName + "/testcases.yml")
          Test.allTestsIndexedByFullName[test.fullName.lower()] = test
        except Exception as e:
          print("ERROR registering test: " + dirName, file=sys.stderr)
          traceback.print_exc()
          sys.exit(1)

  # Runs this test
  #   flavor - "debug" or "release"
  #   device - "cpu" or "gpu"
  #   pyVersion - Python version used (empty string for non-Python tests)
  #   args - command line arguments from argparse
  # returns an instance of TestRunResult
  def run(self, flavor, device, pyVersion, args):
    # measuring the time of running of the test
    startTime = time.time()
    result = self.runImpl(flavor, device, pyVersion, args)
    result.duration = time.time() - startTime
    return result

  def runImpl(self, flavor, device, pyVersion, args):
    result = TestRunResult()
    result.succeeded = True

    # Preparation for pattern-based test cases
    if len(self.testCases) > 0:
      # Locating and reading baseline file
      baselineFile = self.findBaselineFile(flavor, device)
      if baselineFile != None:
          with open(baselineFile, "r") as f:
            baseline = f.read().split("\n")
            if args.verbose:
               print("Baseline: " + baselineFile)

          # Before running the test, pre-creating TestCaseRunResult object for each test case
          # and compute filtered lines from baseline file.
          # Note: some test cases might fail at this time if baseline and/or patterns are inconsistent
          if not args.update_baseline:
            for testCase in self.testCases:
              testCaseRunResult = testCase.processBaseline(baseline)
              if not testCaseRunResult.succeeded:
                 result.succeeded = False
              result.testCaseRunResults.append(testCaseRunResult)
      else:
          if not args.create_baseline:
              return TestRunResult.fatalError("Baseline file sanity check", "Can't find baseline file")

    # preparing run directory
    pyVersionLabel = "_{0}".format(pyVersion) if pyVersion else ''
    runDir = os.path.join(args.run_dir, "{0}_{1}@{2}_{3}{4}".format(self.suite, self.name, flavor, device, pyVersionLabel))
    runDir = cygpath(runDir)
    if not os.path.isdir(runDir):
      os.makedirs(runDir)

    # preparing environment for the test script
    os.environ["TEST_FLAVOR"] = flavor
    os.environ["TEST_DEVICE"] = device
    os.environ["TEST_TAG"] = args.tag or ''
    os.environ["TEST_BUILD_LOCATION"] = args.build_location
    if windows:
      if args.build_sku == "cpu":
        os.environ["TEST_CNTK_BINARY"] = os.path.join(args.build_location, (flavor + "_CpuOnly"), "cntk.exe")
      elif args.build_sku == "uwp":
        os.environ["TEST_CNTK_BINARY"] = os.path.join(args.build_location, (flavor + "_UWP"), "cntk.exe")
      else:
        os.environ["TEST_CNTK_BINARY"] = os.path.join(args.build_location, flavor, "cntk.exe")
      os.environ["MPI_BINARY"] = os.path.join(os.environ["MSMPI_BIN"], "mpiexec.exe")
    else:
      # No UWP on Linux
      assert args.build_sku != "uwp"

      tempPath = os.path.join(args.build_location, args.build_sku, flavor, "bin", "cntk")
      if not os.path.isfile(tempPath):
        for bsku in ["/build/gpu/", "/build/cpu/"]:
          if tempPath.find(bsku) >= 0:
            tempPath = tempPath.replace(bsku, "/build/")
            break
      os.environ["TEST_CNTK_BINARY"] = tempPath
      os.environ["MPI_BINARY"] = "mpiexec"
    # N.B. no cntk.exe in UWP build
    if args.build_sku != "uwp" and not os.path.exists(os.environ["TEST_CNTK_BINARY"]):
      raise ValueError("the cntk executable does not exist at path '%s'"%os.environ["TEST_CNTK_BINARY"])
    os.environ["TEST_BIN_DIR"] = os.path.dirname(os.environ["TEST_CNTK_BINARY"])
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
      print(self.fullName + ":>" + logFile)
    with open(logFile, "w") as output:
      if not windows:
        testScript = self.testDir + "/run-test"
        st = os.stat(testScript)
        os.chmod(testScript, st.st_mode | stat.S_IEXEC | stat.S_IXOTH)
      cmdLine = ["bash", "-c", self.testDir + "/run-test 2>&1"]
      process = subprocess.Popen(cmdLine, stdout=subprocess.PIPE)

      while True:
        line = process.stdout.readline()
        if not line:
           break

        if len(line)>0 and line[-1]=='\n':
          line=line[:len(line)-1]

        if args.verbose:
          # TODO find a better way
          if sys.version_info.major < 3:
            print(self.fullName + ": " + line)
          else:
            print(self.fullName + ": " + line.decode('utf-8').rstrip())

        if args.dry_run:
          print(line)
          continue

        print(line, file=output)
        allLines.append(line)
        output.flush()
        for testCaseRunResult in result.testCaseRunResults:
          testCaseRunResult.testCase.processLine(line, testCaseRunResult, args.verbose)

    exitCode = process.wait()
    success = True

    # saving log file path, so it can be reported later
    result.logFile = logFile

    # checking exit code
    if exitCode != 0:
      if args.dry_run:
        print("[SKIPPED]")
        return result
      else:
        return TestRunResult.fatalError("Exit code must be 0", "==> got exit code {0} when running: {1}".format(exitCode, " ".join(cmdLine)), logFile = logFile)

    # finalizing verification - need to check whether we have any unmatched lines
    for testCaseRunResult in result.testCaseRunResults:
      testCaseRunResult.testCase.finalize(testCaseRunResult)
      if not testCaseRunResult.succeeded:
        result.succeeded = False

    if len(self.testCases)>0 and (args.update_baseline or (baselineFile == None)) and result.succeeded:
      # When running in --update-baseline or --create-baseline mode
      # verifying that new output is successfully matching every pattern in the testcases.yml
      # If this is not the case then baseline update/create will be rejected
      for testCase in self.testCases:
        testCaseRunResult = testCase.processBaseline(allLines)
        if not testCaseRunResult.succeeded:
          result.succeeded = False
        result.testCaseRunResults.append(testCaseRunResult)

      if baselineFile == None:
          baselineFile = self.newBaselineFilePath(device)

      if result.succeeded:
        if args.verbose:
          if args.update_baseline:
            print("Updating baseline file " + baselineFile)
          else:
            print("Creating baseline file " + baselineFile)

        with open(baselineFile, "w") as f:
          f.write("\n".join(allLines))

    return result

  # Composes the full path of a new baseline file for a specified platform and device
  # Note this currently hardcodes the baseline file name to be baseline.<os>.<device>.txt
  # which is how the baselines currently exist for all CNTK E2E tests i.e. we have different
  # baselines for platform and device but not for flavor
  def newBaselineFilePath(self, device):
      candidateName = "baseline" + "." + ("windows" if windows else "linux") + "." + device.lower() + ".txt"
      fullPath = os.path.join(self.testDir, candidateName)
      return fullPath

  # Baseline filename fragments to match for GPU device, decreasing priority.
  gpuBaselinePatternList = []

  # Generate baseline pattern fragments for GPU tests based on the minimum CC
  # level of the cards available on the host. The CC level of only a few cards
  # is made known to the test driver, based on our test lab set up. Note that
  # we don't mix cards of different CC levels.
  # TODO make more general, provide GPU selection
  def getGpuBaselinePatternList(self):
    if not self.gpuBaselinePatternList:
      self.gpuBaselinePatternList = [".gpu", ""]
      if windows:
        nvidiaSmiPath = '/cygdrive/c/Program Files/NVIDIA Corporation/NVSMI/nvidia-smi.exe'
      else:
        nvidiaSmiPath = 'nvidia-smi'
      ccMajorByCard = {
        'GeForce GTX 780 Ti': 3,
        'GeForce GTX 960': 5,
        'GeForce GTX 1050 Ti': 6,
        'Quadro K2000' : 3,
        'Quadro M2000M': 5,
        'Quadro M4000': 5,
        'Tesla M60' : 5,
      }
      cc = sys.maxsize
      try:
        gpuList = subprocess.check_output([nvidiaSmiPath, '-L'])
        for line in gpuList.decode('utf-8').split('\n'):
          m = re.match(r"GPU (?P<id>\d+): (?P<type>[^(]*) \(UUID: (?P<guid>GPU-.*)\)\r?$", line)
          if m:
            try:
              current = ccMajorByCard[m.group('type')]
              if current < cc:
                cc = current
            except KeyError:
              pass
      except OSError:
        pass
      if cc != sys.maxsize:
        self.gpuBaselinePatternList.insert(0, ".gpu.cc" + str(cc))

    return self.gpuBaselinePatternList

  # Finds a location of a baseline file by probing different names in the following order:
  #   baseline.$os.$flavor.$device.txt
  #   baseline.$os.$flavor.txt
  #   baseline.$os.$device.txt
  #   baseline.$os.txt
  #   baseline.$flavor.$device.txt
  #   baseline.$flavor.txt
  #   baseline.$device.txt
  #   baseline.txt
  def findBaselineFile(self, flavor, device):
    if device == 'gpu':
      deviceList = self.getGpuBaselinePatternList()
    else:
      deviceList = ["." + device.lower(), ""]

    for o in ["." + ("windows" if windows else "linux"), ""]:
      for f in ["." + flavor.lower(), ""]:
        for d in deviceList:
          candidateName = "baseline" + o + f + d + ".txt"
          fullPath = cygpath(os.path.join(self.testDir, candidateName), relative=True)
          if os.path.isfile(fullPath):
            return fullPath
    return None

  # Checks whether the test matches the specified tag,
  # returns matched tag name on success, or None if there is no match(boolean, string) tuple
  def matchesTag(self, tag, flavor, device, os, build_sku):
    tagL = tag.lower() # normalizing the tag for comparison
    # enumerating all the tags
    for tag in list(self.tags.keys()):
      # match by direct string comparison or by prefix matching rule:
      # e.g: 'bvt' matches 'bvt' 'bvt-a', 'bvt-b' but not 'bvtx'
      if tag==tagL or tag.startswith(tagL + "-"):
        # evaluating tag's predicate
        if self.tags[tag](flavor=flavor, device=device, os=os, build_sku=build_sku):
          return tag
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
          print("ERROR registering pattern: " + pattern, file=sys.stderr)
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
        # we have matched line in the output and at least one remaining unmatched in a baseline
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
            print("[FAILED]: Testcase " + self.name)
            print("Baseline: " + expected)

          # also show all failed patterns
          for p in failedPatterns:
            msg = "Failed pattern: " + p.patternText
            if verbose:
              print(msg)
            result.diagnostics+=msg+"\n"
        # removing this line, since we already matched it (whether successfully or not - doesn't matter)
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

    # Transforming our pattern into a single regular expression
    # processing {{...}} fragments and escaping all regex special characters
    self.regexText = prefix + re.sub(r"(\{\{[^}]+\}\}|[\[\]\.\*\+\{\}\(\)\$\^\\\|\?])", self.patternParse, patternText)
    # Compiling it to perform a check (fail-fast) and for faster matching later
    self.regex = re.compile(self.regexText)


  # this is a callback method passed to re.sub call above - it performs the core parsing logic
  def patternParse(self, match):
    fragment = match.group(1)
    if len(fragment) == 1:
      # this is a special character of regex
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
      # converting this to regex which matches specific type
      # All {{...}} sections are converted to regex groups named as G0, G1, G2...
      return "(?P<G{0}>{1})".format(len(self.groupInfo)-1, TestPattern.typeTable[dataType])

  # Checks whether given line matches this pattern
  # returns True or False
  def match(self, line):
    if type(line) == bytes:
      line = line.decode("utf-8")
    return self.regex.match(line) != None

  # Compares a line from baseline log and a line from real output against this pattern
  # return true or false
  def compare(self, expected, actual):
    if type(actual) == bytes:
      actual = actual.decode("utf-8")
    #import pdb;pdb.set_trace()
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
    self.duration = -1

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

def enumerateCommandLineTests(args):
  """Enumerates tests on the command line."""
  if not args.test:
    return

  if len(args.test) == 1 and args.test[0].startswith('@'):
    filename = os.path.realpath(args.test[0][1:])
    if not os.path.isfile(filename):
      print("ERROR: file not found", filename, file=sys.stderr)
      sys.exit(1)

    with open(filename) as f:
      for line in f.readlines():
        line = line.strip()
        if not line:
          continue
       
        if line.startswith('#'):
          continue

        yield line.rstrip('/')

    return

  for arg in args.test:
    yield arg.rstrip('/')

# Lists all available tests
def listCommand(args):
  testsByTag = {}
  args.test = [ arg.lower() for arg in enumerateCommandLineTests(args) ]

  for test in list(Test.allTestsIndexedByFullName.values()):
     if not args.test or test.fullName.lower() in args.test:
        for flavor in args.flavors:
           for device in args.devices:
              for operating_system in args.oses:
                for build_sku in args.buildSKUs:
                  if build_sku=="cpu" and device=="gpu":
                    continue
                  tag = test.matchesTag(args.tag, flavor, device, operating_system, build_sku) if args.tag else '*'
                  if tag:
                    if tag in list(testsByTag.keys()):
                      testsByTag[tag].add(test.fullName)
                    else:
                      testsByTag[tag] = set([test.fullName])
  for tag in sorted(testsByTag.keys()):
    if tag=="*":
      print(' \n'.join(sorted(testsByTag[tag])))
    else:
      print(tag + ": " + ' '.join(sorted(testsByTag[tag])))

# Runs given test(s) or all tests
def runCommand(args):
  testsToRun = []

  for arg in enumerateCommandLineTests(args):
    arg_lower = arg.lower()

    if arg_lower not in Test.allTestsIndexedByFullName:
      print("ERROR: test not found", arg, file=sys.stderr)
      sys.exit(1)

    testsToRun.append(Test.allTestsIndexedByFullName[arg_lower])

  if not testsToRun:
    testsToRun = list(sorted(Test.allTestsIndexedByFullName.values(), key=lambda test: test.fullName))
  
  devices = args.devices
  flavors = args.flavors

  convertPythonPath = lambda path: os.pathsep.join([cygpath(y) for y in path.split(',')])
  pyPaths = {}
  if args.py27_paths:
    pyPaths['py27'] = convertPythonPath(args.py27_paths)
  if args.py34_paths:
    pyPaths['py34'] = convertPythonPath(args.py34_paths)
  if args.py35_paths:
    pyPaths['py35'] = convertPythonPath(args.py35_paths)
  if args.py36_paths:
    pyPaths['py36'] = convertPythonPath(args.py36_paths)
  # If no Python was explicitly specified, go against current.
  if not pyPaths:
    pyPaths['py'] = ''

  # Remember original PATH environment variable
  originalPath = os.environ["PATH"]

  os.environ["TEST_ROOT_DIR"] = os.path.dirname(os.path.realpath(sys.argv[0]))

  print("CNTK Test Driver is started")
  print("Running tests:  " + " ".join([y.fullName for y in testsToRun]))
  print("Build location: " + args.build_location)
  print("Build SKU:      " + args.build_sku)
  print("Run location:   " + args.run_dir)
  print("Flavors:        " + " ".join(flavors))
  print("Devices:        " + " ".join(devices))
  if (args.update_baseline):
    print("*** Running in automatic baseline update mode ***")
  print("")
  if args.dry_run:
    os.environ["DRY_RUN"] = "1"
  succeededCount, totalCount = 0, 0
  for test in testsToRun:
    for flavor in flavors:
      for device in devices:
        for build_sku in args.buildSKUs:
          testPyPaths = pyPaths if test.isPythonTest else {'': ''}

          for pyVersion in sorted(testPyPaths.keys()):
            if args.tag and args.tag != '' and not test.matchesTag(args.tag, flavor, device, 'windows' if windows else 'linux', build_sku):
              continue
            if build_sku=="cpu" and device=="gpu":
              continue
            totalCount = totalCount + 1
            if len(test.testCases)==0:
              # forcing verbose mode (showing all output) for all test which are based on exit code (no pattern-based test cases)
              args.verbose = True

            pyTestLabel = " {0}".format(pyVersion) if pyVersion else ''

            if testPyPaths[pyVersion]:
              os.environ["PATH"] = testPyPaths[pyVersion] + os.pathsep + originalPath

            # Printing the test which is about to run (without terminating the line)
            sys.stdout.write("Running test {0} ({1} {2}{3}) - ".format(test.fullName, flavor, device, pyTestLabel));
            if args.dry_run:
              print("[SKIPPED] (dry-run)")
              continue
              
            # in verbose mode, terminate the line, since there will be a lot of output
            if args.verbose:
              sys.stdout.write("\n");
            sys.stdout.flush()
            # Running the test and collecting a run results
            result = test.run(flavor, device, pyVersion, args)

            if args.verbose:
              # writing the test name one more time (after possibly long verbose output)
              sys.stdout.write("Test finished {0} ({1} {2}{3}) - ".format(test.fullName, flavor, device, pyTestLabel));
            if result.succeeded:
              succeededCount = succeededCount + 1
              # in no-verbose mode this will be printed in the same line as 'Running test...'
              print("[OK] {0:.2f} sec".format(result.duration))
            else:
              print("[FAILED] {0:.2f} sec".format(result.duration))
            # Showing per-test-case results:
            for testCaseRunResult in result.testCaseRunResults:
              if testCaseRunResult.succeeded:
                # Printing 'OK' test cases only in verbose mode
                if (args.verbose):
                  print(" [OK] " + testCaseRunResult.testCaseName)
              else:
                # 'FAILED' + detailed diagnostics with proper indentation
                print(" [FAILED] " + testCaseRunResult.testCaseName)
                if testCaseRunResult.diagnostics:
                  for line in testCaseRunResult.diagnostics.split('\n'):
                    print("    " + line);
                # In non-verbose mode log wasn't piped to the stdout, showing log file path for convenience

            # Restore original path
            os.environ["PATH"] = originalPath

            if not result.succeeded and not args.verbose and result.logFile:
              print("  See log file for details: " + result.logFile)

  if args.dry_run:
    sys.exit(1)
    
  if args.update_baseline:
    print("{0}/{1} baselines updated, {2} failed".format(succeededCount, totalCount, totalCount - succeededCount))
  else:
    print("{0}/{1} tests passed, {2} failed".format(succeededCount, totalCount, totalCount - succeededCount))
  if succeededCount != totalCount:
    sys.exit(10)

# ======================= Entry point =======================
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="TestDriver - CNTK Test Driver")
  subparsers = parser.add_subparsers(help="command to execute. Run TestDriver.py <command> --help for command-specific help")
  runSubparser = subparsers.add_parser("run", help="run test(s)")
  runSubparser.add_argument("test", nargs="*",
                            help="optional test name(s) to run, specified as Suite/TestName. "
                                 "Use list command to list available tests. "
                                 "If not specified then all tests will be run."
                                 " "
                                 "'@<filename>' can be used to specify a line-delimited list "
                                 "of tests.")

  defaultBuildSKU = "gpu"

  runSubparser.add_argument("-b", "--build-location", help="location of the CNTK build to run")
  runSubparser.add_argument("-t", "--tag", help="runs tests which match the specified tag")
  runSubparser.add_argument("-d", "--device", help="cpu|gpu - run on a specified device")
  runSubparser.add_argument("-f", "--flavor", help="release|debug - run only a specified flavor")
  runSubparser.add_argument("-s", "--build-sku", default=defaultBuildSKU, help="cpu|gpu - run tests only for a specified build SKU")
  runSubparser.add_argument("--py27-paths", help="comma-separated paths to prepend when running a test against Python 2.7")
  runSubparser.add_argument("--py34-paths", help="comma-separated paths to prepend when running a test against Python 3.4")
  runSubparser.add_argument("--py35-paths", help="comma-separated paths to prepend when running a test against Python 3.5")
  runSubparser.add_argument("--py36-paths", help="comma-separated paths to prepend when running a test against Python 3.6")
  tmpDir = os.getenv("TEMP") if windows else "/tmp"
  defaultRunDir = os.path.join(tmpDir, "cntk-test-{0}.{1}".format(time.strftime("%Y%m%d%H%M%S"), random.randint(0,1000000)))
  runSubparser.add_argument("-r", "--run-dir", default=defaultRunDir, help="directory where to store test output, default: a random dir within /tmp")
  runSubparser.add_argument("--update-baseline", action='store_true', help="update baseline file(s) instead of matching them")
  runSubparser.add_argument("--create-baseline", action='store_true', help="create new baseline file(s) (named as baseline.<os>.<device>.txt) for tests that do not currently have baselines")
  runSubparser.add_argument("-v", "--verbose", action='store_true', help="verbose output - dump all output of test script")
  runSubparser.add_argument("-n", "--dry-run", action='store_true', help="do not run the tests, only print test names and configurations to be run along with full command lines")

  runSubparser.set_defaults(func=runCommand)

  listSubparser = subparsers.add_parser("list", help="list available tests")
  listSubparser.add_argument("-t", "--tag", help="limits a resulting list to tests matching the specified tag")
  listSubparser.add_argument("-d", "--device", help="cpu|gpu - tests for a specified device")
  listSubparser.add_argument("-f", "--flavor", help="release|debug - tests for specified flavor")
  listSubparser.add_argument("-s", "--build-sku", default=defaultBuildSKU, help="cpu|gpu - list tests only for a specified build SKU")
  listSubparser.add_argument("--os", help="windows|linux - tests for a specified operating system")
  listSubparser.add_argument("test", nargs="*",
                             help="optional test name(s) to list, specified as Suite/TestName. "
                                  " "
                                  "'@<filename>' can be used to specify a line-delimited list "
                                  "of tests.")

  listSubparser.set_defaults(func=listCommand)

  if len(sys.argv)==1:
      parser.print_help()
      sys.exit(1)

  args = parser.parse_args(sys.argv[1:])

  # parsing a --device, --flavor and --os options:
  args.devices = ["cpu", "gpu"]
  if (args.device):
    args.device = args.device.lower()
    if not args.device in args.devices:
      print("--device must be one of", args.devices, file=sys.stderr)
      sys.exit(1)
    args.devices = [args.device]

  args.flavors = ["debug", "release"]
  if (args.flavor):
    args.flavor = args.flavor.lower()
    if not args.flavor in args.flavors:
      print("--flavor must be one of", args.flavors, file=sys.stderr)
      sys.exit(1)
    args.flavors = [args.flavor]

  args.buildSKUs = ["cpu", "gpu", "uwp"]
  if (args.build_sku):
    args.build_sku = args.build_sku.lower()
    if not args.build_sku in args.buildSKUs:
      print("--build-sku must be one of", args.buildSKUs, file=sys.stderr)
      sys.exit(1)
    args.buildSKUs = [args.build_sku]
    if (args.build_sku == "cpu" or args.build_sku == "uwp") and args.devices == ["gpu"]:
      print("Invalid combination: --build-sku cpu and --device gpu", file=sys.stderr)
      sys.exit(1)

  if args.func == runCommand and not args.build_location:
    args.build_location = os.path.realpath(os.path.join(thisDir, "../..", "x64" if windows else "build/"))

  if args.func == listCommand:
    args.oses = ["windows", "linux"]
    if (args.os):
      args.os = args.os.lower()
      if not args.os in args.oses:
        print("--os must be one of", args.oses, file=sys.stderr)
        sys.exit(1)
    args.oses = [args.os]

  # discover all the tests
  Test.discoverAllTests()

  # execute the command
  args.func(args)

