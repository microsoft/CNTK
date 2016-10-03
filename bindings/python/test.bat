setlocal

set PATH=%cd%\..\..\x64\Release;%PATH%
set PYTHONPATH=%cd%;%cd%\examples;%PYTHONPATH%

cd cntk\ops\tests
echo RUNNING cntk\ops unit tests...
pytest
echo(
cd ..\..\..

cd examples\CifarResNet
echo RUNNING Cifar ResNet example...
python CifarResNet.py
echo(
cd ..\..

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
