import os, sys, importlib
import shutil, time
import subprocess
import PARAMETERS
locals().update(importlib.import_module("PARAMETERS").__dict__)


####################################
# Parameters
####################################
cntkCmdStrPattern = "cntk.exe configFile={0}configbs.cntk currentDirectory={0} "


####################################
# Main
####################################
# copy config file
shutil.copy(cntkTemplateDir + "configbs.cntk", cntkFilesDir)

# run cntk
tstart = datetime.datetime.now()
os.environ['ACML_FMA'] = str(0)
cmdStr = cntkCmdStrPattern.format(cntkFilesDir)
print cmdStr
pid = subprocess.Popen(cmdStr, cwd = cntkFilesDir)
pid.wait()
print ("Time running cntk [s]: " + str((datetime.datetime.now() - tstart).total_seconds()))

# delete model files written during cntk training
modelDir = cntkFilesDir + "Output/"
filenames = getFilesInDirectory(modelDir, postfix = None)
for filename in filenames:
    if not filename.endswith('Fast-RCNN'):
        os.remove(modelDir + filename)
assert pid.returncode == 0, "ERROR: cntk ended with exit code {}".format(pid.returncode)

print "DONE."