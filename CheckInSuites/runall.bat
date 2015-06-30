::: this runs all tests in this folder
::: BUGBUG: so far only the ASR tests have updated pathnames etc., the others are non-functional stubs here that need to be updated
::: TODO: find a good solution for specifying directories for data that we cannot distribute with CNTK ourselves.

set BUILD=%1

set  THIS=%~dp0

::: ASR tests
::: BUGBUG: We do not get to see stdout from CNTK, only from the BAT files.
( %THIS%\ASR\config\runall.bat cpu %BUILD% ) 2>&1
( %THIS%\ASR\config\runall.bat gpu %BUILD% ) 2>&1

::: LM tests
::: TODO: provide BAT file

::: MNIST
::: TODO: provide BAT file

::: SLU
::: TODO: update paths
C:\dev\cntk3\CNTKSolution\x64\Release\cntk configFile=globals.config+rnnlu.config
