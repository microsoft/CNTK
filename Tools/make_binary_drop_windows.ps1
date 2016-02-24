# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

# WARNING. This will run in Microsoft Internal Environment ONLY

# Get Jenkins environment Variables
$buildConfig =(Get-Item env:"BUILD_CONFIGURATION").Value
$targetConfig =(Get-Item env:"TARGET_CONFIGURATION").Value

# If not a Release build quit
If ($buildConfig -ne "Release")
{
	Write-Host "Not a release build. No binary drops generaion"
	Exit
}

# Set Paths
$basePath = "BinaryDrops\ToZip"
$baseDropPath = $basePath + "\cntk"
$zipFile = "BinaryDrops\BinaryDrops.zip"
If ($targetConfig -eq "CPU")
{
	$buildPath = "x64\Release_CpuOnly"
}
Else
{
	$buildPath = "x64\Release"
}
$sharePath = "\\muc-vfs-01a\CNTKshare\CNTK-Binary-Drop" + "\" + $buildConfig


# Make binary drop folder
New-Item -Path $baseDropPath -ItemType "directory"

# Copy build binaries
Copy-Item $buildPath -Recurse -Destination $baseDropPath\cntk

# Clean unwanted items
If (Test-Path $baseDropPath\cntk\UnitTests) {Remove-Item $baseDropPath\cntk\UnitTests -Recurse}
Remove-Item $baseDropPath\cntk\*test*.exe
Remove-Item $baseDropPath\cntk\*.pdb
Remove-Item $baseDropPath\cntk\*.lib
Remove-Item $baseDropPath\cntk\*.exp
Remove-Item $baseDropPath\cntk\*.metagen

# Copy Examples
Copy-Item Examples -Recurse -Destination $baseDropPath\Examples

# Copy all items from the share
Copy-Item $sharePath\*  -Recurse -Destination $baseDropPath\cntk


# Make ZIP file
$source = $PWD.Path + "\" + $basePath
Write-Host $source
$destination = $PWD.Path + "\" + $zipFile
Write-Host $destination
Add-Type -assembly "system.io.compression.filesystem"
[io.compression.zipfile]::CreateFromDirectory($source, $destination)

# Remove ZIP sources
If (Test-Path $basePath) {Remove-Item $basePath -Recurse}
