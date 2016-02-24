# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.


# CNTK files

# ACML_PATH + lib
# CUDA_PATH + bin
# CUDNN_PATH +bin
# OPENCV_PATH + x64\vc12\bin
# C:\jenkins\workspace\CNTK-Build-Windows\x64\Release

# MPI file

# ACML files

# CUDA files

# cuDNN files

# NVML file

# $test="C:\CNTK-Test\Source"
# if (-not $test.EndsWith("\"))
# {
#	$test = $test + "\"
# }

# Write-Host $test

# $source = "C:\CNTK-Test\Source"
# $destination = "C:\CNTK-Test\Backup.zip"
# If(Test-path $destination) {Remove-item $destination}

# Add-Type -assembly "system.io.compression.filesystem"

# [io.compression.zipfile]::CreateFromDirectory($Source, $destination)

# $x = "Path"
# (get-item env:$x).Value
# Write-Host $x

$basePath = "BinaryDrops\ToZip"
$baseDropPath = $basePath + "\cntk"
$zipFile = "BinaryDrops\BinaryDrops.zip"

# Make dir structure
# New-Item -Path 'BinaryDrops\cntk\cntk' -ItemType "directory"
New-Item -Path $baseDropPath"\license" -ItemType "directory"
New-Item -Path $baseDropPath"\prerequisites" -ItemType "directory"
# New-Item -Path 'BinaryDrops\cntk\Examples' -ItemType "directory"

# Copy build binaries
Copy-Item x64\Release -Recurse -Destination $baseDropPath\cntk

# Clean unwanted items
If (Test-Path $baseDropPath\cntk\UnitTests) {Remove-Item $baseDropPath\cntk\UnitTests -Recurse}
Remove-Item $baseDropPath\cntk\*test*.exe
Remove-Item $baseDropPath\cntk\*.pdb
Remove-Item $baseDropPath\cntk\*.lib
Remove-Item $baseDropPath\cntk\*.exp
Remove-Item $baseDropPath\cntk\*.metagen

# Copy Examples
Copy-Item Examples -Recurse -Destination $baseDropPath\Examples

# TODO Copy Lisence etc

# Make ZIP file
$source = $PWD.Path + "\" + $basePath
Write-Host $source
$destination = $PWD.Path + "\" + $zipFile
Write-Host $destination
Add-Type -assembly "system.io.compression.filesystem"
[io.compression.zipfile]::CreateFromDirectory($source, $destination)

# Remove ZIP sources
If (Test-Path $basePath) {Remove-Item $basePath -Recurse}