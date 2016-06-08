#!/usr/bin/env python
# ----------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# ---------------------------------------------------------
# This is a test driver for running end-to-end CNTK tests

import sys, os, csv, traceback, re

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

class Baseline:
  def __init__(self, fullPath, opSystem, device, flavor, testResult = "", trainResult = ""):
    self.fullPath = fullPath
    self.opSystem = opSystem
    self.device = device
    self.flavor = flavor
    self.testResult = testResult
    self.trainResult = trainResult

  def getOsDeviceFlavor(self):
    return "-".join([self.opSystem, self.device, self.flavor])

  def getResultsInfo(self, baselineContent):
    trainResults = re.findall('.*(Finished Epoch\[[ ]*\d+ of \d+\]\: \[Training\]) (.*)', baselineContent, re.MULTILINE)        
    if trainResults:                                       
      self.trainResult = Baseline.getLastTrainResult(trainResults[-1])[0:-2]
    testResults = re.findall('.*(Final Results: Minibatch\[1-\d+\]:(\s+\* \d+|))\s+(.*)', baselineContent, re.MULTILINE)
    if testResults:
      self.testResult = Baseline.getLastTestResult(testResults[-1])[0:-2]

  @staticmethod
  def getLastTestResult(line):
    return line[0] + line[1] + "\n" + line[2].replace('; ', '\n').replace('    ','\n')

  @staticmethod
  def getLastTrainResult(line):  
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
    self.cpuInfo = ""  
    self.gpuInfo = ""

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
            baseline = Baseline(fullPath, o[1:], device[1:], flavor[1:]);            
            baselineFilesList.append(baseline)

    return baselineFilesList

  def getCpuInfo(self, baselineContent):
    cpuInfo = re.search(".*Hardware info:\s+"
					"CPU Model (Name:\s*.*)\s+"
					"CPU (Cores:\s*.*)\s+"
					"(Hardware threads: \d+)\s+"
					"Total (Memory:\s*.*)\s+"
					"GPU Model (Name: .*)?\s+"
					"GPU (Memory: .*)?", baselineContent)
    if cpuInfo is None:
      return
    self.cpuInfo = "\n".join(cpuInfo.groups()[0:4])
    hwInfo = cpuInfo.groups()[4:len(cpuInfo.groups())]

    gpuInfoIndex = baselineContent.find("GPU info: ")
    gpuInfo = re.findall("\t\t(Device ID: \d+)\s+"
    "(Compute Capability: \d\.\d)\s+"
    "(CUDA cores: \d+)", baselineContent[gpuInfoIndex:gpuInfoIndex+1500]) #Taking just a portion of the file, in order to avoid wasting time searching it all
    if not gpuInfo:
      return
    for index in range(0, len(gpuInfo)):
      hwInfo = hwInfo + gpuInfo[index]
    self.gpuInfo = "\n".join(hwInfo)  


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
        gitHash = re.search('.*Build SHA1:\s([a-z0-9]{40})\s', baselineContent)
        if gitHash is None:
          continue
        example.gitHash = gitHash.group(1) 
        example.getCpuInfo(baselineContent)                 
        baseline.getResultsInfo(baselineContent)                 
      example.baselineList.append(baseline)    
        
def writeMetricsToCsvFile():
  metricsFile = open("metrics.csv",'wb')
  csvWriter = csv.writer(metricsFile, dialect='excel', quoting=csv.QUOTE_ALL)
  tableHeader = ['Example', 'Git Hash', 'Hardware', 'GPU', 'Log file', 'OS - Device - Flavor', 'Train Result', 'Test Result']
  csvWriter.writerow(tableHeader)

  for example in Example.allExamplesIndexedByFullName: 
    firstBaseline = example.baselineList[0]    
      
    csvWriter.writerow([example.fullName, example.gitHash, example.cpuInfo, example.gpuInfo, firstBaseline.fullPath.split(thisDir)[1][1:], firstBaseline.getOsDeviceFlavor(), firstBaseline.trainResult, firstBaseline.testResult])  
    for baseline in example.baselineList[1:]:
      csvWriter.writerow(['', '', '', '', baseline.fullPath.split(thisDir)[1][1:], baseline.getOsDeviceFlavor(), baseline.trainResult, baseline.testResult])

def createMarkdownExampleList(file):
  for example in Example.allExamplesIndexedByFullName:
    if not example.baselineList:
      continue
    file.write("".join(["* [", example.fullName, "](#", example.fullName, ")\n"]))
  file.write("\n")

def writeMetricsToMarkdown():
  metricsFile = open("metrics.md",'wb')

  createMarkdownExampleList(metricsFile)
  
  for example in Example.allExamplesIndexedByFullName:
    if not example.baselineList:
      continue

    lineBreak = "  \n"
    metricsFile.write("".join(["**Example:** ", "<a name=\"", example.fullName, "\"></a>", example.fullName, lineBreak]))
    metricsFile.write("".join(["**Git Hash:** ", example.gitHash, lineBreak]))    
    metricsFile.write("".join(["**CPU:** ", example.cpuInfo.replace('\n', '. ')  , lineBreak]))
    metricsFile.write("".join(["**GPU:** ", example.gpuInfo.replace('\n', '. ')  , lineBreak]))

    metricsFile.write('\n|Log file |OS - Device - Flavor | Train Result | Test Result|\n')
    metricsFile.write('|---|---|---|---|\n')
    for baseline in example.baselineList:
      pipeChar = "|"
      
      metricsFile.write("".join(['|[', baseline.fullPath.split("/")[-1], "](../blob/master/Tests/EndToEndTests/", 
                        baseline.fullPath.split(thisDir)[1][1:], ")|", baseline.getOsDeviceFlavor(), pipeChar, baseline.trainResult.replace("\n", " "), pipeChar,  baseline.testResult.replace("\n", " "), pipeChar, "\n"]))

  metricsFile.write("\n")

# ======================= Entry point =======================
six.print_("==============================================================================")

Example.discoverAllExamples()

getExamplesMetrics()

#writeMetricsToCsvFile()

writeMetricsToMarkdown()

