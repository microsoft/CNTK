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

pushd ..\..\Tests\EndToEndTests\CNTKv2Python\Examples
echo RUNNING cntk\Tests\EndToEndTests\CNTKv2Python\Examples tests
pytest
echo(
popd


endlocal
