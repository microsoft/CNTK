@echo off
setlocal enabledelayedexpansion

REM Copyright (c) Microsoft. All rights reserved.
REM
REM Licensed under the MIT license. See LICENSE.md file in the project root
REM for full license information.
REM ==============================================================================

REM Grab the parameters
REM
REM Note: don't rely on environment variables, since properties may have been
REM overridden at msbuild invocation.

set p_OutDir=%~1
set p_DebugBuild=%~2
set p_GpuBuild=%~3
set p_CNTK_VERSION=%~4
shift
set p_CNTK_VERSION_BANNER=%~4
shift
set p_CNTK_COMPONENT_VERSION=%~4
set p_SWIG_PATH=%~5
set p_CNTK_PYTHON_WITH_DEPS=%~6
shift
set p_CNTK_PY_VERSIONS=%~6
set p_CNTK_PY27_PATH=%~7
set p_CNTK_PY34_PATH=%~8
set p_CNTK_PY35_PATH=%~9
shift
set p_CNTK_PY36_PATH=%~9

REM Construct p_CNTK_PY_VERSIONS if not explicitly defined
REM (Note: to disable Python build completely, no CNTK_PYx_PATH variable must be defined)
if not defined p_CNTK_PY_VERSIONS (
  REM Note: leading space doesn't hurt
  if defined p_CNTK_PY27_PATH set p_CNTK_PY_VERSIONS=!p_CNTK_PY_VERSIONS! 27
  if defined p_CNTK_PY34_PATH set p_CNTK_PY_VERSIONS=!p_CNTK_PY_VERSIONS! 34
  if defined p_CNTK_PY35_PATH set p_CNTK_PY_VERSIONS=!p_CNTK_PY_VERSIONS! 35
  if defined p_CNTK_PY36_PATH set p_CNTK_PY_VERSIONS=!p_CNTK_PY_VERSIONS! 36
)

REM Validate p_CNTK_PY_VERSIONS contents.
for %%p in (%p_CNTK_PY_VERSIONS%) do (
  if not "%%~p" == "27" if not "%%~p" == "34" if not "%%~p" == "35" if not "%%~p" == "36" echo Build for unsupported Python version '%%~p' requested, stopping&exit /b 1
)

REM Validate p_CNTK_PY_VERSIONS contents.
REM (Note: Don't merge with above loop; more robust parsing)
set nothingToBuild=1
for %%p in (%p_CNTK_PY_VERSIONS%) do (
  call set extraPath=!p_CNTK_PY%%~p_PATH!
  if not defined extraPath echo Build for Python version '%%~p' requested, but CNTK_PY%%~p_PATH not defined, stopping&exit /b 1
  set nothingToBuild=
)
if defined nothingToBuild echo Python support not configured to build.&exit /b 0

if "%p_DebugBuild%" == "true" echo Currently no Python build for Debug configurations, exiting.&exit /b 0

if not defined VS2017INSTALLDIR (
  @echo Environment variable VS2017INSTALLDIR not defined.
  @echo Make sure Visual Studion 2017 is installed.
  exit /b 1
)

REM vcvarsall.bat scripts changes current working directory,
REM   so we need to save it and restore it afterwards
pushd .
if not exist "%VS2017INSTALLDIR%\VC\Auxiliary\build\vcvarsall.bat" (
  echo Error: "%VS2017INSTALLDIR%\VC\Auxiliary\build\vcvarsall.bat" not found.
  echo Make sure you have installed Visual Studion 2017 correctly.
  exit /b 1
)
call "%VS2017INSTALLDIR%\VC\Auxiliary\build\vcvarsall.bat" amd64 -vcvars_ver=14.11
popd

set CNTK_LIB_PATH=%p_OutDir%

set DIST_DIR=%p_OutDir%\Python
set PATH=%p_SWIG_PATH%;%PATH%
set CNTK_VERSION=%p_CNTK_VERSION%
set CNTK_VERSION_BANNER=%p_CNTK_VERSION_BANNER%
set CNTK_COMPONENT_VERSION=%p_CNTK_COMPONENT_VERSION%
set MSSdk=1
set DISTUTILS_USE_SDK=1

pushd "%CNTK_LIB_PATH%"
if errorlevel 1 echo Cannot change directory.&exit /b 1

set CNTK_LIBRARIES=
for %%D in (
  Cntk.Composite-%CNTK_COMPONENT_VERSION%.dll
  Cntk.Core-%CNTK_COMPONENT_VERSION%.dll
  Cntk.Deserializers.Binary-%CNTK_COMPONENT_VERSION%.dll
  Cntk.Deserializers.HTK-%CNTK_COMPONENT_VERSION%.dll
  Cntk.Deserializers.TextFormat-%CNTK_COMPONENT_VERSION%.dll
  Cntk.Math-%CNTK_COMPONENT_VERSION%.dll
  Cntk.ExtensibilityExamples-%CNTK_COMPONENT_VERSION%.dll  
  Cntk.PerformanceProfiler-%CNTK_COMPONENT_VERSION%.dll
  Cntk.DelayLoadedExtensions-%CNTK_COMPONENT_VERSION%.dll
  libiomp5md.dll
  mklml.dll
) do (
  if defined CNTK_LIBRARIES (
    set CNTK_LIBRARIES=!CNTK_LIBRARIES!;%CNTK_LIB_PATH%\%%D
  ) else (
    set CNTK_LIBRARIES=%CNTK_LIB_PATH%\%%D
  )
)

@REM mkldnn.dll is optional
if exist mkldnn.dll (
 set CNTK_LIBRARIES=!CNTK_LIBRARIES!;%CNTK_LIB_PATH%\mkldnn.dll
)

@REM Cntk.BinaryConvolution-%CNTK_COMPONENT_VERSION%.dll is optional
if exist Cntk.BinaryConvolution-%CNTK_COMPONENT_VERSION%.dll (
 set CNTK_LIBRARIES=!CNTK_LIBRARIES!;%CNTK_LIB_PATH%\Cntk.BinaryConvolution-%CNTK_COMPONENT_VERSION%.dll
)

@REM Cntk.Deserializers.Image-%CNTK_COMPONENT_VERSION%.dll (plus dependencies) is optional
if exist Cntk.Deserializers.Image-%CNTK_COMPONENT_VERSION%.dll for %%D in (
  Cntk.Deserializers.Image-%CNTK_COMPONENT_VERSION%.dll
  opencv_world*.dll
  zip.dll
  zlib.dll
) do set CNTK_LIBRARIES=!CNTK_LIBRARIES!;%CNTK_LIB_PATH%\%%D

if /i %p_GpuBuild% equ true for %%D in (
  cublas64_*.dll
  cudart64_*.dll
  cudnn64_*.dll
  curand64_*.dll
  cusparse64_*.dll
  nvml.dll
) do (
  set CNTK_LIBRARIES=!CNTK_LIBRARIES!;%CNTK_LIB_PATH%\%%D
)

set PYTHON_PROJECT_NAME=cntk
if /i %p_GpuBuild% equ true (
  set PYTHON_PROJECT_NAME=cntk-gpu
)

set PYTHON_WITH_DEPS=
if /i %p_CNTK_PYTHON_WITH_DEPS% equ true (
  set PYTHON_WITH_DEPS="--with-deps"
)

popd
if errorlevel 1 echo Cannot restore directory.&exit /b 1

REM Build everything in supplied order
set oldPath=%PATH%
for %%p in (%p_CNTK_PY_VERSIONS%) do (
  call set extraPath=!p_CNTK_PY%%~p_PATH!
  echo Building for Python version '%%~p', extra path is !extraPath!
  set PATH=!extraPath!;!oldPath!
  python.exe .\setup.py --project-name %PYTHON_PROJECT_NAME% %PYTHON_WITH_DEPS% ^
      build_ext --inplace --force --compiler msvc --plat-name=win-amd64 ^
      bdist_wheel --dist-dir "%DIST_DIR%"
  if errorlevel 1 exit /b 1
)
