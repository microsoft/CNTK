setlocal

cd "%~dp0\cntk" && ^
pytest --deviceid gpu && ^
set PYTHONPATH=%CD%;%PYTHONPATH% && ^
pushd ..\..\..\Tests\EndToEndTests\CNTKv2Python\Examples && ^
pytest --deviceid gpu

@REM N.B. running the examples above doesn't exactly mirror our
@REM      test in CI, where each of the examples runs in its own process.
@REM      Here, test may fail due to test setup influencing each other.
