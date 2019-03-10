#!/usr/bin/env python
# ----------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
# ---------------------------------------------------------
# This script extracts information (hardware used, final results) contained in the baselines files
# and generates a markdown file (wiki page)

import sys, os, re
import TestDriver as td

try:
  import six
except ImportError:
  print("Python package 'six' not installed. Please run 'pip install six'.")
  sys.exit(1)

thisDir = os.path.dirname(os.path.realpath(__file__))
windows = os.getenv("OS")=="Windows_NT"

class Baseline:
  def __init__(self, fullPath, testResult = "", trainResult = ""):
    self.fullPath = fullPath
    self.cpuInfo = ""
    self.gpuInfo = ""
    self.testResult = testResult
    self.trainResult = trainResult

  # extracts results info. e.g.
  # Finished Epoch[ 5 of 5]: [Training] ce = 2.32253198 * 1000 err = 0.90000000 * 1000 totalSamplesSeen = 5000 learningRatePerSample = 2e-06 epochTime=0.175781
  # Final Results: Minibatch[1-1]: err = 0.90000000 * 100 ce = 2.32170486 * 100 perplexity = 10.1930372
  def extractResultsInfo(self, baselineContent):
    trainResults = re.findall(r'.*(Finished Epoch\[ *\d+ of \d+\]\: \[Training\]) (.*)', baselineContent)
    if trainResults:                                       
      self.trainResult = Baseline.formatLastTrainResult(trainResults[-1])[0:-2]
    testResults = re.findall(r'.*(Final Results: Minibatch\[1-\d+\]:)(\s+\* \d+)?\s+(.*)', baselineContent)
    if testResults:
      self.testResult = Baseline.formatLastTestResult(testResults[-1])[0:-2]

  # extracts cpu and gpu info from baseline content. e.g.:
  #CPU info:
  #  CPU Model Name: Intel(R) Xeon(R) CPU E5-2620 v3 @ 2.40GHz
  #  Hardware threads: 12
  #GPU info:
  #
  #Device[0]: cores = 2496; computeCapability = 5.2; type = "Quadro M4000"; memory = 8192 MB
  #Device[1]: cores = 96; computeCapability = 2.1; type = "Quadro 600"; memory = 1024 MB
  #  Total Memory: 33474872 kB
  def extractHardwareInfo(self, baselineContent):
    startCpuInfoIndex = baselineContent.find("CPU info:")
    endCpuInfoIndex = baselineContent.find("----------", startCpuInfoIndex)
    cpuInfo = re.search(r"^CPU info:\s+"
                        r"CPU Model (Name:\s*.*)\s+"
                        r"(Hardware threads: \d+)\s+"
                        r"Total (Memory:\s*.*)\s+", baselineContent[startCpuInfoIndex:endCpuInfoIndex], re.MULTILINE)
    if cpuInfo is None:
      return
    self.cpuInfo = "\n".join(cpuInfo.groups())

    startGpuInfoIndex = baselineContent.find("GPU info:")
    endGpuInfoIndex = baselineContent.find("----------", startGpuInfoIndex)
    gpuInfoSnippet = baselineContent[startGpuInfoIndex:endGpuInfoIndex]

    gpuDevices = re.findall(r"\t\t(Device\[\d+\]: cores = \d+; computeCapability = \d\.\d; type = .*; memory = \d+ MB)[\r\n]?", gpuInfoSnippet)
    if not gpuDevices:
      return
    gpuInfo = [ device for device in gpuDevices ]
    self.gpuInfo = "\n".join(gpuInfo)

  @staticmethod
  def formatLastTestResult(line):
    return line[0] + line[1] + "\n" + line[2].replace('; ', '\n').replace('    ','\n')

  @staticmethod
  def formatLastTrainResult(line):
    epochsInfo, parameters = line[0], line[1]
    return epochsInfo + '\n' + parameters.replace('; ', '\n')

class Example:

  allExamplesIndexedByFullName = {} 

  def __init__(self, suite, name, testDir):
    self.suite = suite
    self.name = name
    self.fullName = suite + "/" + name
    self.testDir = testDir
    self.baselineList = []
    
    self.gitHash = ""

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

        example = Example(suiteName,  exampleName, testDir)
        Example.allExamplesIndexedByFullName[example.fullName.lower()] = example

  # it returns a list with all baseline files for current example
  def findBaselineFilesList(self):
    baselineFilesList = []

    oses = [".windows", ".linux", ""]
    devices = [".cpu", ".gpu", ""]
    flavors = [".debug", ".release", ""]

    for o in oses:
      for device in devices:
        for flavor in flavors:          
          candidateName = "baseline" + o + flavor + device + ".txt"
          fullPath = td.cygpath(os.path.join(self.testDir, candidateName), relative=True)          
          if os.path.isfile(fullPath):
            baseline = Baseline(fullPath);
            baselineFilesList.append(baseline)

    return baselineFilesList

# extracts information for every example and stores it in Example.allExamplesIndexedByFullName
def getExamplesMetrics():  
  Example.allExamplesIndexedByFullName = list(sorted(Example.allExamplesIndexedByFullName.values(), key=lambda test: test.fullName))  
  allExamples = Example.allExamplesIndexedByFullName

  print ("CNTK - Metrics collector")  

  for example in allExamples:
    baselineListForExample = example.findBaselineFilesList() 
    six.print_("Example: " + example.fullName)   
    for baseline in baselineListForExample:        
      with open(baseline.fullPath, "r") as f:
        baselineContent = f.read()
        gitHash = re.search(r'.*Build SHA1:\s([a-z0-9]{40})[\r\n]+', baselineContent, re.MULTILINE)
        if gitHash is None:
          continue
        example.gitHash = gitHash.group(1) 
        baseline.extractHardwareInfo(baselineContent)
        baseline.extractResultsInfo(baselineContent)
      example.baselineList.append(baseline)    
        
# creates a list with links to each example result
def createAsciidocExampleList(file):
  for example in Example.allExamplesIndexedByFullName:
    if not example.baselineList:
      continue
    file.write("".join(["<<", example.fullName.replace("/","").lower(),",", example.fullName, ">> +\n"]))
  file.write("\n")

def writeMetricsToAsciidoc():
  metricsFile = open("metrics.adoc",'wb')

  createAsciidocExampleList(metricsFile)

  for example in Example.allExamplesIndexedByFullName:
    if not example.baselineList:
      continue
    metricsFile.write("".join(["===== ", example.fullName, "\n"]))
    metricsFile.write("".join(["**Git Hash: **", example.gitHash, "\n\n"]))
    metricsFile.write("[cols=3, options=\"header\"]\n")
    metricsFile.write("|====\n")
    metricsFile.write("|Log file / Configuration | Train Result | Test Result\n")
    for baseline in example.baselineList:
      pathInDir=baseline.fullPath.split(thisDir)[1][1:]
      metricsFile.write("".join(["|link:../blob/", example.gitHash[:7],"/Tests/EndToEndTests/", pathInDir, "[",
                                 baseline.fullPath.split("/")[-1], "] .2+|", baseline.trainResult.replace("\n", " "), " .2+|",
                                 baseline.testResult.replace("\n", " "), "|\n"]))
      cpuInfo = "".join(["CPU: ", re.sub("[\r]?\n", ' ', baseline.cpuInfo)])

      gpuInfo = re.sub("[\r]?\n", ' ', baseline.gpuInfo)
      if gpuInfo:
        metricsFile.write("".join([cpuInfo, " GPU: ", gpuInfo]))
      else:
        metricsFile.write(cpuInfo)

    metricsFile.write("\n|====\n\n")

# ======================= Entry point =======================
six.print_("==============================================================================")

Example.discoverAllExamples()

getExamplesMetrics()

writeMetricsToAsciidoc()
