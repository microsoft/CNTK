setlocal

cd "%~dp0"

set PATH=%CD%\..\..\x64\Release;%PATH%
set PYTHONPATH=%CD%;%CD%\examples;%PYTHONPATH%

pushd cntk\tests
echo RUNNING cntk unit tests...
pytest --deviceid gpu
echo(
popd

pushd cntk\ops\tests
echo RUNNING cntk\ops unit tests...
pytest
echo(
popd

pushd examples\CifarResNet
echo RUNNING Cifar ResNet example...
python CifarResNet.py
echo(
popd

echo RUNNING MNIST feed-forward classifier example...
python examples\MNIST\SimpleMNIST.py
echo(

echo RUNNING feed-forward numpy interop example...
python examples\NumpyInterop\FeedForwardNet.py
echo(

echo RUNNING sequence-to-sequence example...
python examples\Sequence2Sequence\Sequence2Sequence.py
echo(

echo RUNNING sequence classification example...
python examples\SequenceClassification\SequenceClassification.py
echo(

endlocal
