#!/usr/bin/env python

# This script collects coverage for all unit tests in a specified directory.
# For each test file outputs a binary coverage file produces by the VS tool chain.
# The coverage file can be opened in VS for detailed analysis or be used for reporting.
# For collection of full coverage the binaries should be compiled with /PROFILE linker flag.

import os
import argparse
import subprocess

def collectCoverage(tests, testDir, outputDir, toolDir, config):
    coverage = os.path.join(toolDir, "CodeCoverage.exe")

    for test in tests:
        outputFile = os.path.join(outputDir, test + ".coverage")
        print "Running executable %s with result in %s" % (test, outputFile)
        subprocess.check_call([coverage, "collect", "/output:%s" % outputFile, "" if config == "" else "/config:%s" % config, os.path.join(testDir, test)])

def collectCoverageSingle(test, outputDir, toolDir, config):    
    collectCoverage([os.path.basename(test)], os.path.dirname(test), outputDir, toolDir, config)

def collectCoverageMulti(testDir, outputDir, toolDir, config):
    tests = [ f for f in os.listdir(testDir) if os.path.isfile(os.path.join(testDir, f)) and f.lower().endswith(".exe") ]
    collectCoverage(tests, testDir, outputDir, toolDir, config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collects coverage for the executable or directory with test executables")
    parser.add_argument('--test', help='Path to the executable or directory that has to be analyzed', required=True)
    parser.add_argument('--outputdir', help='Output directory for coverage results', required=True)
    parser.add_argument('--tooldir', help='Tool directory for CodeCoverage tool', required=False, default=r'c:\Program Files (x86)\Microsoft Visual Studio 14.0\Team Tools\Dynamic Code Coverage Tools\amd64')
    parser.add_argument('--config', help='Configuration for CodeCoverage tool', required=False, default="")   

    args = parser.parse_args()

    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)

    if os.path.isfile(args.test):
        collectCoverageSingle(args.test, args.outputdir, args.tooldir, args.config)
    elif os.path.isdir(args.test):
        collectCoverageMulti(args.test, args.outputdir, args.tooldir, args.config)
    else:
        print('Please specify correct executable or test directory where the coverage should be collected.')
