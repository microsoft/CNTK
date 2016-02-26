# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

# WARNING. This will run in Microsoft Internal Environment ONLY
# Generating CNTK Binary drops in Jenkins environment

# Enable Verbose automatically
[CmdletBinding()]
param ()

# Set to Stop on Error
$ErrorActionPreference = 'Stop'

# Set Verbose mode
if ($verbose)
{
	 $VerbosePreference = "continue"
}

Write-Verbose "Making binary drops..."

# Get Jenkins environment Variables
$buildConfig =(Get-Item env:BUILD_CONFIGURATION).Value
$targetConfig =(Get-Item env:TARGET_CONFIGURATION).Value

# If not a Release build quit
If ($buildConfig -ne "Release")
{
	Write-Verbose "Not a release build. No binary drops generation"
	Exit
}

# Set Paths
$basePath = "BinaryDrops\ToZip"
$baseDropPath = Join-Path $basePath -ChildPath cntk
$zipFile = "BinaryDrops\BinaryDrops.zip"
$buildPath = "x64\Release"
If ($targetConfig -eq "CPU")
{
	$buildPath = "x64\Release_CpuOnly"
}
$sharePath = Join-Path "\\muc-vfs-01a\CNTKshare\CNTK-Binary-Drop" -ChildPath $targetConfig


# Make binary drop folder
New-Item -Path $baseDropPath -ItemType "directory"

# Copy build binaries
Write-Verbose "Copying build binaries ..."
Copy-Item $buildPath -Recurse -Destination $baseDropPath\cntk

# Clean unwanted items
If (Test-Path $baseDropPath\cntk\UnitTests) {Remove-Item $baseDropPath\cntk\UnitTests -Recurse}
Remove-Item $baseDropPath\cntk\*test*.exe
Remove-Item $baseDropPath\cntk\*.pdb
Remove-Item $baseDropPath\cntk\*.lib
Remove-Item $baseDropPath\cntk\*.exp
Remove-Item $baseDropPath\cntk\*.metagen

# Copy Examples
Write-Verbose "Copying Examples ..."
Copy-Item Examples -Recurse -Destination $baseDropPath\Examples

# Copy all items from the share
# For whatever reason Copy-Item in the line below does not work
# Copy-Item $sharePath"\*"  -Recurse -Destination $baseDropPath
# Copying with Robocopy
Write-Verbose "Copying dependencies and other files from Remote Share ..."
robocopy $sharePath $baseDropPath /s /e

Write-Verbose "Making ZIP and cleaning up..."

# Make ZIP file
$source = Join-Path $PWD.Path -ChildPath $basePath
$destination = Join-Path $PWD.Path -ChildPath $zipFile
Add-Type -assembly "system.io.compression.filesystem"
[io.compression.zipfile]::CreateFromDirectory($source, $destination)

# Remove ZIP sources
If (Test-Path $basePath) {Remove-Item $basePath -Recurse}
