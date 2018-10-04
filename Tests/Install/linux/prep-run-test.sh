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

# onnx_model_test requires onnx to be installed.
# Skip this test until we decide to add onnx dependencies to OOBE test environment. 
[ "\$TEST_DEVICE" = "gpu" ] && pytest "\$MODULE_DIR" --deviceid \$TEST_DEVICE -k "not onnx_model_test"
# TODO not all (doc) tests run on CPU

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

if [ "\$TEST_DEVICE" = "gpu" ]; then
  cd "$CNTK_DROP/Tutorials"
  for f in *.ipynb; do
    # TODO 203 fails when run without GUI?
    # 204: interactive
    # 104: occasional "ZeroDivisionError: integer division or modulo by zero" before training - data download issue?
    if [[ \$f != CNTK_203_Reinforcement_Learning_Basics.ipynb && \$f != CNTK_204_Sequence_To_Sequence.ipynb && \$f != CNTK_104_Finance_Timeseries_Basic_with_Pandas_Numpy.ipynb ]]; then
      jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=python\$(python -c "import sys; print(sys.version_info[0])") --ExecutePreprocessor.timeout=2700 --output \$(basename \$f .ipynb)-out.ipynb \$f
    fi
  done
fi

# CNTK.wiki example:
cd "$CNTK_DROP/Tutorials/HelloWorld-LogisticRegression"
cntk configFile=lr_bs.cntk deviceId=\$TEST_DEVICE_ID

cd "$CNTK_DROP/Examples/Image/GettingStarted"
cntk configFile=01_OneHidden.cntk deviceId=\$TEST_DEVICE_ID

# Example with image deserializer
cd "$CNTK_DROP/Examples/Image/Regression"
cntk configFile=RegrSimple_CIFAR10.cntk deviceId=\$TEST_DEVICE_ID

RUNTEST

chmod 755 $RUN_TEST
chown testuser:testuser $RUN_TEST
