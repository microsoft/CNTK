#!/usr/bin/env python
# ----------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# ---------------------------------------------------------
# This is a test driver for running end-to-end CNTK tests

import sys, os, traceback, subprocess, random, re, time, stat

try:
  import six
except ImportError:
  print("Python package 'six' not installed. Please run 'pip install six'.")
  sys.exit(1)

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

# This class encapsulates an instance of the example
class Example:
  # "Suite/TestName" => instance of Test
  allExamplesIndexedByFullName = {} 

  def __init__(self, suite, name, testDir):
    self.suite = suite
    self.name = name
    self.fullName = suite + "/" + name
    self.testDir = testDir
    self.testResult = ""
    self.trainResult = ""

  # Populates Tests.allTestsIndexedByFullName by scanning directory tree
  @staticmethod
  def discoverAllExamples():
    testsDir = thisDir
    for dirName, subdirList, fileList in os.walk(testsDir):
      if 'testcases.yml' in fileList:
        testDir = dirName
        exampleName = os.path.basename(dirName)
        suiteDir = os.path.dirname(dirName)
        # suite name will be derived from the path components
        suiteName = os.path.relpath(suiteDir, testsDir).replace('\\', '/')        

        #if suiteName.startswith("Examples"):
        example = Example(suiteName,  exampleName, testDir)
        Example.allExamplesIndexedByFullName[example.fullName.lower()] = example

  # Finds a location of a baseline file by probing different names in the following order:
  #   baseline.$os.$flavor.$device.txt
  #   baseline.$os.$flavor.txt
  #   baseline.$os.$device.txt
  #   baseline.$os.txt
  #   baseline.$flavor.$device.txt
  #   baseline.$flavor.txt
  #   baseline.$device.txt
  #   baseline.txt
  def findBaselineFilesList(self):
    baselineFilesList = []

    oses = [".windows", ".linux", ""]
    devices = [".cpu", ".gpu", ""]
    flavors = [".debug", ".release", ""]

    for o in oses:
      for device in devices:
        for flavor in flavors:          
          candidateName = "baseline" + o + flavor + device + ".txt"
          fullPath = cygpath(os.path.join(self.testDir, candidateName), relative=True)          
          if os.path.isfile(fullPath):            
            baselineFilesList.append(fullPath)

    return baselineFilesList

def getLastTestResult(line):
  return line[0] + line[1] + "\n" + line[2].replace('; ', '\n').replace('    ','\n')

def getLastTrainResult(line):
  separator = "\n[Training]\n"
  epochsInfo, parameters = line[0], line[1]  
  return epochsInfo + separator + parameters.replace('; ', '\n')

def runCommand():
  Example.allExamplesIndexedByFullName = list(sorted(Example.allExamplesIndexedByFullName.values(), key=lambda test: test.fullName))
  allExamples = Example.allExamplesIndexedByFullName

  print ("CNTK - Metrics collector")
  six.print_("Getting examples:  " + " ".join([y.fullName for y in allExamples]))

  for example in allExamples:    
    baselineListForExample = example.findBaselineFilesList()  
    for baseline in baselineListForExample:      
      with open(baseline, "r") as f:
        baselineContent = f.read()        
        trainResults = re.findall('.*(Finished Epoch\[[ ]*\d+ of \d+\]\:) \[Training\] (.*)', baselineContent, re.MULTILINE)
        testResults = re.findall('.*(Final Results: Minibatch\[1-\d+\]:(\s+\* \d+|))\s+(.*)', baselineContent, re.MULTILINE)        
        if trainResults:
          six.print_("==============================================================================")
          six.print_("Suite Name " + example.suite)
          six.print_("Example " + example.name)
          six.print_("Baseline: " + baseline + "\n")                    
          six.print_(getLastTrainResult(trainResults[-1]))
          six.print_("")
          if testResults:
            six.print_(getLastTestResult(testResults[-1]))
          gitHash = re.search('.*Build SHA1:\s([a-z0-9]{40})\s', baselineContent)
          if gitHash is not None:
            six.print_("\nBuild Hash: ")
            six.print_(gitHash.group(1))
          hardwareInfo = re.search(".*Hardware info:\s+"
					"CPU Model Mame:\s*(.*)\s+"
					"CPU cores:\s*(.*)\s+"
					"Hardware threads: (\d+)\s+"
					"Total Memory:\s*(.*)\s+"
					"GPU Model Name: (.*)?\s+"
					"GPU Memory: (.*)?", baselineContent)
          if hardwareInfo is not None:
            six.print_("Hardware information information: ")
            six.print_("CPU model " + hardwareInfo.groups(1))
            six.print_("CPU cores " + hardwareInfo.groups(2))
            six.print_("Hardware threads: " + hardwareInfo.groups(3))          
            six.print_("Total memory: " + hardwareInfo.groups(4))
            six.print_("GPU name: " + hardwareInfo.groups(5))
            six.print_("GPU name: " + hardwareInfo.groups(6))
          gpuInfo = re.search(".*GPU info:\s+Device ID: (\d+)\s+"
          "Compute Capability: (\d\.\d)\s+"
          "CUDA cores: (\d+)", baselineContent)
          if gpuInfo is not None:
            six.print_("Additional GPU information: ")
            six.print_("CPU model " + hardwareInfo.groups(1))
            six.print_("CPU cores " + hardwareInfo.groups(2))
            six.print_("Hardware threads: " + hardwareInfo.groups(3)) 

six.print_("==============================================================================")
        
# ======================= Entry point =======================
# discover all the tests
Example.discoverAllExamples()

runCommand()