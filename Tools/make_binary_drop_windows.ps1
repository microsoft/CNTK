#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#
[CmdletBinding()]
param(
    [parameter(Mandatory=$true)][string]$targetConfig,           # the config created (CPU, GPU, ...)
    [parameter(Mandatory=$true)][string]$targetConfigSuffix,     # the config suffix (CPU-Only, GPU, GPU-1bit-SGD ...)
    [parameter(Mandatory=$true)][string]$releaseTag,             # the tag of the release (2-0-beta11-0)
    [parameter(Mandatory=$true)][string]$commit,
    [parameter(Mandatory=$true)][string]$outputFileName,         # the generated zip file name
    [parameter(Mandatory=$true)][string]$sharePath)

$ErrorActionPreference = 'Stop'

$workSpace = $PWD.Path
Write-Verbose "Making binary drops..."

# Set Paths
$basePath = "ToZip"
$baseDropPath = Join-Path $basePath -ChildPath cntk
$baseIncludePath = Join-Path $baseDropPath -ChildPath Include
$buildPath = "x64\Release"
if ($targetConfig -eq "CPU")
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

# binaryDrop locations
$artifactPath = Join-Path $workSpace BinaryDrops
$whlArtifactFolder = Join-Path $artifactPath $targetConfigSuffix
New-Item -Path $artifactPath -ItemType directory -force
New-Item -Path $whlArtifactFolder -ItemType directory -force

# Copy wheels to destination
Copy-Item $buildPath\Python\*.whl $whlArtifactFolder

# Make binary drop folder
New-Item -Path $baseDropPath -ItemType directory

# create version.txt file
$fileContent = "CNTK-{0}`r`nRelease`r`n{1}`r`n{2}`r`n" -f $releaseTag, $targetConfigSuffix, $commit
$fileContent | Set-Content -Encoding Ascii $baseDropPath\version.txt

# Copy build binaries
Write-Verbose "Copying build binaries ..."
Copy-Item $buildPath -Recurse -Destination $baseDropPath\cntk

# Clean unwanted items
Remove-Item $baseDropPath\cntk\*test*.exe*
Remove-Item $baseDropPath\cntk\*.pdb
Remove-Item $baseDropPath\cntk\python -Recurse

# Keep EvalDll.lib
Remove-Item $baseDropPath\cntk\*.lib  -Exclude EvalDll.lib, CNTKLibrary-2.0.lib
Remove-Item $baseDropPath\cntk\*.exp
Remove-Item $baseDropPath\cntk\*.metagen
# Remove specific items
Remove-Item $baseDropPath\cntk\CommandEval.exe -Force -ErrorAction SilentlyContinue
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
Remove-Item $baseDropPath\Scripts\pytest.ini -Force -ErrorAction SilentlyContinue
Remove-Item -Recurse $baseDropPath\Scripts\install\linux -Force -ErrorAction SilentlyContinue

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
# Switched to use 7zip because of the backslash separator issue in .NET compressor
# (fixed in 4.6.1, which is not a standard component of build machines
# see https://msdn.microsoft.com/en-us/library/mt712573(v=vs.110).aspx?f=255&MSPPError=-2147217396 )
$source = Join-Path $workSpace -ChildPath $basePath
$destination = Join-Path $artifactPath -ChildPath $outputFileName
Set-Location -Path $source
7za a -bd $destination .
If ($LastExitCode -ne 0)
{
    throw "7za returned exit code $LastExitCode"
}

Set-Location -Path $workSpace

# Log the file hash
Get-FileHash -Algorithm SHA256 -Path $destination, $whlArtifactFolder\*.whl

# Remove ZIP sources
Remove-Item -Recurse $basePath -Force -ErrorAction SilentlyContinue 

exit 0
