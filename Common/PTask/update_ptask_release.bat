@echo off

if not exist version.txt (
  echo version.txt not found in current directory. Please run from \Common\PTask in your CNTK tree. Exiting.
  goto:end
)

if [%DANDELION_ROOT%] == [] (
  echo DANDELION_ROOT environment variable must be set. Exiting.
  goto:end
)

echo Checking out existing PTask files...
tf checkout . /r

echo Copying PTask release artifacts from %DANDELION_ROOT% ...
copy /Y %DANDELION_ROOT%\ptask\ptask\bin\x64\Release\ptask.lib lib\Release\ptask.lib
copy /Y %DANDELION_ROOT%\ptask\ptask\bin\x64\Release\ptask.pdb lib\Release\ptask.pdb
copy /Y %DANDELION_ROOT%\ptask\ptask\bin\x64\Debug\ptask.lib lib\Debug\ptask.lib
copy /Y %DANDELION_ROOT%\ptask\ptask\bin\x64\Debug\ptask.pdb lib\Debug\ptask.pdb
copy /Y %DANDELION_ROOT%\ptask\ptask\*.h include

echo Making sure any new files are added to the repository...
tf add . /r
echo.
echo ** Safe to ignore any warnings above about items already having pending changes **

echo.
echo Once you are ready to check in an update to PTask, perfrom the following steps:
echo.
echo // Note a timestamp that the PTask repository could be rolled back to to re-build this version of PTask.
echo notepad version.txt
echo.
echo // Perform checkin - automatically omits any files identical to their latest checked in version.
echo tf checkin
echo.
echo // View the contents of the checkin.
echo tf changeset nnnnn
echo.
echo // Check which files are still checked out.
echo tf status . /r
echo.
echo // Revert any files still checked out.
echo tf undo . /r
:end
