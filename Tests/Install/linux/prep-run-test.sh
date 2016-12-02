#!/bin/bash
# TODO nvidia-smi to check availability of GPUs for GPU tests

CNTK_DROP=\$HOME/cntk

RUN_TEST=/home/testuser/run-test.sh
cat >| $RUN_TEST <<RUNTEST
set -e -x
TEST_DEVICE=\$1
export PATH="$PATH"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
source ~/cntk/activate-cntk

# Just for informational purposes:
[ "\$TEST_DEVICE" = "gpu" ] && nvidia-smi

TEST_DEVICE_ID=-1
[ "\$TEST_DEVICE" = "gpu" ] && TEST_DEVICE_ID=0

which cntk
MODULE_DIR="\$(python -c "import cntk, os, sys; sys.stdout.write(os.path.dirname(os.path.abspath(cntk.__file__)))")"
[ \$? -eq 0 ]
pytest "\$MODULE_DIR" --deviceid \$TEST_DEVICE --doctest-modules

# Installation validation example from CNTK.wiki (try from two different paths):
cd "$CNTK_DROP/Tutorials"

python NumpyInterop/FeedForwardNet.py
cd NumpyInterop
python FeedForwardNet.py

cd "$CNTK_DROP/Examples/Image/DataSets/MNIST"
python install_mnist.py

cd "$CNTK_DROP/Examples/Image/DataSets/CIFAR-10"
python install_cifar10.py

# TODO run some examples

# TODO actually do different device and syntax.

# CNTK.wiki example:
cd "$CNTK_DROP/Tutorials/HelloWorld-LogisticRegression"
cntk configFile=lr_bs.cntk deviceId=\$TEST_DEVICE_ID

cd "$CNTK_DROP/Examples/Image/GettingStarted"
cntk configFile=01_OneHidden.cntk deviceId=\$TEST_DEVICE_ID

RUNTEST

chmod 755 $RUN_TEST
chown testuser:testuser $RUN_TEST
