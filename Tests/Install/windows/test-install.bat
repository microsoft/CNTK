@REM ==============================================================================
@REM Copyright (c) Microsoft. All rights reserved.
@REM Licensed under the MIT license. See LICENSE.md file in the project root
@REM for full license information.
@REM ==============================================================================

setlocal

@REM Trick cntkpy35.bat into believing we're legitimate
set CMDCMDLINE="%COMSPEC%" &@REM do not delete the space to the left

call "%~1"

for /f "delims=" %%i in ('python -c "import cntk, os, sys; sys.stdout.write(os.path.dirname(os.path.abspath(cntk.__file__)))"') do set MODULE_DIR=%%i
if errorlevel 1 exit /b 1

where cntk && ^
pytest %MODULE_DIR% && ^
cd cntk && ^
cd Tutorials && ^
python NumpyInterop/FeedForwardNet.py && ^
python NumpyInterop\FeedForwardNet.py && ^
cd NumpyInterop && ^
python FeedForwardNet.py && ^
cd ..\HelloWorld-LogisticRegression && ^
cntk configFile=lr_bs.cntk && ^
cd ..\..\Examples\Image && ^
cd DataSets/MNIST && ^
python install_mnist.py && ^
cd ..\..\GettingStarted && ^
cntk configFile=01_OneHidden.cntk

exit /b %ERRORLEVEL%
