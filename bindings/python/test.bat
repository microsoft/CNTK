setlocal

cd "%~dp0"

set PATH=%CD%\..\..\x64\Release;%PATH%
set PYTHONPATH=%CD%;%CD%\examples;%PYTHONPATH%

pushd cntk\tests
echo RUNNING cntk unit tests...
pytest --deviceid gpu
echo(
popd

pushd cntk\debugging\tests
echo RUNNING cntk\debugging unit tests...
pytest
echo(
popd

pushd cntk\internal\tests
echo RUNNING cntk\internal unit tests...
pytest
echo(
popd

pushd cntk\io\tests
echo RUNNING cntk\io unit tests...
pytest
echo(
popd

pushd cntk\layers\tests
echo RUNNING cntk\layers unit tests...
pytest
echo(
popd

pushd cntk\learners\tests
echo RUNNING cntk\learners unit tests...
pytest
echo(
popd

pushd cntk\logging\tests
echo RUNNING cntk\logging unit tests...
pytest
echo(
popd

pushd cntk\losses\tests
echo RUNNING cntk\losses unit tests...
pytest
echo(
popd

pushd cntk\metrics\tests
echo RUNNING cntk\metrics unit tests...
pytest
echo(
popd

pushd cntk\ops\tests
echo RUNNING cntk\ops unit tests...
pytest
echo(
popd

pushd cntk\train\tests
echo RUNNING cntk\train unit tests...
pytest
echo(
popd

pushd cntk\utils\tests
echo RUNNING cntk\utils unit tests...
pytest
echo(
popd

pushd ..\..\Tests\EndToEndTests\CNTKv2Python\Examples
echo RUNNING cntk\Tests\EndToEndTests\CNTKv2Python\Examples tests
pytest
echo(
popd


endlocal
