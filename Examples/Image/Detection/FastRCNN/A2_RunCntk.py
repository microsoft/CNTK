from __future__ import print_function
import os, sys, importlib
import shutil, time
import subprocess
import PARAMETERS
locals().update(importlib.import_module("PARAMETERS").__dict__)


####################################
# Parameters
####################################
cntkCmdStrPattern = "cntk.exe configFile={0}fastrcnn.cntk currentDirectory={0} {1}"

# cntk arguments
NumLabels = nrClasses

NumTrainROIs = cntk_nrRois
TrainROIDim = cntk_nrRois * 4
TrainROILabelDim = cntk_nrRois * nrClasses

NumTestROIs = cntk_nrRois
TestROIDim = cntk_nrRois * 4
TestROILabelDim = cntk_nrRois * nrClasses

cntk_args = "NumLabels={} NumTrainROIs={}".format(NumLabels, NumTrainROIs)
cntk_args += " TrainROIDim={} TrainROILabelDim={}".format(TrainROIDim, TrainROILabelDim)
cntk_args += " NumTestROIs={}".format(NumTestROIs)
cntk_args += " TestROIDim={} TestROILabelDim={}".format(TestROIDim, TestROILabelDim)

####################################
# Main
####################################
# copy config file
shutil.copy(cntkTemplateDir + "fastrcnn.cntk", cntkFilesDir)

# run cntk
tstart = datetime.datetime.now()
os.environ['ACML_FMA'] = str(0)
cmdStr = cntkCmdStrPattern.format(cntkFilesDir, cntk_args)
print (cmdStr)
pid = subprocess.Popen(cmdStr, cwd = cntkFilesDir)
pid.wait()
print ("Time running cntk [s]: " + str((datetime.datetime.now() - tstart).total_seconds()))

# delete intermediate model files written during cntk training
modelDir = cntkFilesDir + "Output/"
filenames = getFilesInDirectory(modelDir, postfix = None)
for filename in filenames:
    if filename.contains('Fast-RCNN.model.'):
        os.remove(modelDir + filename)
assert pid.returncode == 0, "ERROR: cntk ended with exit code {}".format(pid.returncode)

print ("DONE.")