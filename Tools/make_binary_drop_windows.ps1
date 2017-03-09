# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

# WARNING. This will run in Microsoft Internal Environment ONLY
# Generating CNTK Binary drops in Jenkins environment

# Command line parameters
# Verbose command line parameter (-verbose) is automatically added
# because of CmdletBinding
[CmdletBinding()]
param
(
    # Supposed to be taken from Jenkins BUILD_CONFIGURATION
    [string]$buildConfig,

    # Supposed to be taken from Jenkins TARGET_CONFIGURATION
    [string]$targetConfig,

    # File share path. Supposed to have sub-folders corresponding to $targetConfig
    [string]$sharePath,

    # Output file path.
    [string]$outputPath = "BinaryDrops.zip"
)

# Set to Stop on Error
$ErrorActionPreference = 'Stop'

# Manual parameters check rather than using [Parameter(Mandatory=$True)]
# to avoid the risk of interactive prompts inside a Jenkins job
$usage = " parameter is missing. Usage example: make_binary_drop_windows.ps1 -buildConfig Release -targetConfig gpu -sharePath \\server\share"
If (-not $buildConfig) {Throw "buildConfig" + $usage}
If (-not $targetConfig) {Throw "targetConfig" + $usage}
If (-not $sharePath) {Throw "sharePath" + $usage}

# Set Verbose mode
If ($verbose)
{
     $VerbosePreference = "continue"
}

Write-Verbose "Making binary drops..."

# If not a Release build quit
If ($buildConfig -ne "Release")
{
    Write-Verbose "Not a release build. No binary drops generation"
    Exit
}

# Set Paths
$basePath = "BinaryDrops\ToZip"
$baseDropPath = Join-Path $basePath -ChildPath cntk
$baseIncludePath = Join-Path $baseDropPath -ChildPath Include
$buildPath = "x64\Release"
If ($targetConfig -eq "CPU")
{
    $buildPath = "x64\Release_CpuOnly"
}
# Include Files
$includePath = "Source\Common\Include"
$includePath20 = "Source\CNTKv2LibraryDll\API"
$includeFiles = New-Object string[] 3
$includeFiles[0] = Join-Path $includePath -ChildPath Eval.h
$includeFiles[1] = Join-Path $includePath20 -ChildPath CNTKLibrary.h
$includeFiles[2] = Join-Path $includePath20 -ChildPath CNTKLibraryInternals.h
$sharePath = Join-Path $sharePath -ChildPath $targetConfig

# Copy Wheels
Copy-Item $buildPath\Python\*.whl

# Make binary drop folder
New-Item -Path $baseDropPath -ItemType directory

# Copy build binaries
Write-Verbose "Copying build binaries ..."
Copy-Item $buildPath -Recurse -Destination $baseDropPath\cntk

# Clean unwanted items
Remove-Item $baseDropPath\cntk\*test*.exe*
Remove-Item $baseDropPath\cntk\*.pdb
# Keep EvalDll.lib
Remove-Item $baseDropPath\cntk\*.lib  -Exclude EvalDll.lib, CNTKLibrary-2.0.lib
Remove-Item $baseDropPath\cntk\*.exp
Remove-Item $baseDropPath\cntk\*.metagen
# Remove specific items
If (Test-Path $baseDropPath\cntk\CommandEval.exe)
{
    Remove-Item $baseDropPath\cntk\CommandEval.exe
}
Remove-Item $baseDropPath\cntk\Microsoft.VisualStudio.QualityTools.UnitTestFramework.*

# Make Include folder
New-Item -Path $baseIncludePath -ItemType directory

# Copy Include
Write-Verbose "Copying Include files ..."
Foreach ($includeFile in $includeFiles)
{
    Copy-Item $includeFile -Destination $baseIncludePath
}

# Copy Examples
Write-Verbose "Copying Examples ..."
Copy-Item Examples -Recurse -Destination $baseDropPath\Examples

# Copy Examples
Write-Verbose "Copying Tutorials ..."
Copy-Item Tutorials -Recurse -Destination $baseDropPath\Tutorials

# Copy Scripts
Write-Verbose "Copying Scripts ..."
Copy-Item Scripts -Recurse -Destination $baseDropPath\Scripts
# Remove some files if they exist
If (Test-Path $baseDropPath\Scripts\pytest.ini)
{
    Remove-Item $baseDropPath\Scripts\pytest.ini
}
If (Test-Path $baseDropPath\Scripts\install\linux)
{
    Remove-Item -Recurse $baseDropPath\Scripts\install\linux
}

# Copy all items from the share
# For whatever reason Copy-Item in the line below does not work
# Copy-Item $sharePath"\*"  -Recurse -Destination $baseDropPath
# Copying with Robocopy. Maximum 2 retries, 30 sec waiting time in between
Write-Verbose "Copying dependencies and other files from Remote Share ..."
robocopy $sharePath $baseDropPath /s /e /r:2 /w:30 /np
# Check that Robocopy finished OK.
# Any exit code greater than 7 indicates error
# See http://ss64.com/nt/robocopy-exit.html
If ($LastExitCode -gt 7)
{
    Throw "Copying from Remote Share failed. Robocopy exit code is " + $LastExitCode
}

Write-Verbose "Making ZIP and cleaning up..."

# Make ZIP file
$source = Join-Path $PWD.Path -ChildPath $basePath
$destination = Join-Path $PWD.Path -ChildPath $outputPath
Add-Type -assembly "system.io.compression.filesystem"
[io.compression.zipfile]::CreateFromDirectory($source, $destination)

# Log the file hash
Get-FileHash -Algorithm SHA256 -Path $destination, *.whl

# Remove ZIP sources
If (Test-Path $basePath)
{
    Remove-Item $basePath -Recurse
}

# Return zero exit code code from here (N.B.: can be non-zero from robocopy above)
exit 0
