#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#

Set-StrictMode -Version Latest

function Get-FreeDiskSpaceGB(
    [Parameter(Mandatory=$True)][string] $driveLetter)
{
    try {
        if ($driveLetter.Length -eq 1) {
            $driveLetter += ":"
        }
        $targetDisk = Split-Path $driveLetter -Qualifier

        $disks = Get-WmiObject win32_Volume
        $disk = $disks | Where-Object { $_.DriveLetter -eq $targetDisk }

        return [int] ($disk.FreeSpace / 1GB)
    }
    catch {
        return [int] 0
    }
}

function Compress-CacheDirectory(
    [Parameter(Mandatory=$True)][string] $directory,
    [int] $maxCount = 48,
    [int] $minFreeDiskSpaceGB = 16,
    [string] $filter = "*.*",
    [switch] $isContainer)
{
    try {
        $selection = @(Get-ChildItem -path $directory -filter $filter | Where-Object { $_.PSIsContainer -eq $isContainer })
        $fileCount = $selection.Count

        if ($fileCount -gt 0) {
            $freeSpaceGB = Get-FreeDiskSpaceGB -driveLetter (Split-Path $directory -Qualifier)
            
    
            if (($fileCount -gt $maxCount) -or ($freeSpaceGB -lt $minFreeDiskSpaceGB)){
                $toSkip = $fileCount - 4
                if ($toSkip -lt 1) {
                    $toSkip = 1
                }
                # TODO: this is not concurrency safe. Another job could use a directory we are trying to remove ...
                $selection | Sort-Object LastWriteTime -Descending | Select-Object -Skip $toSkip | Remove-Item -Recurse -ErrorAction SilentlyContinue
            }
        }
    }
    catch {
        # suppress all errors
    }
}

function Get-TempFileName(
    [string] $filePrefix,
    [string] $tempDir = "")
{
    if (-not $tempDir) {
        $tempDir = [System.IO.Path]::GetTempPath()
    }
    if ($filePrefix) {
        $fileName = [io.path]::GetFileNameWithoutExtension($filePrefix)
    }
    else {
        $fileName = [string]::Empty
    }
    $fileName += ([GUID]::NewGuid()).Guid
    return Join-Path $tempDir $filename
}

function Update-FileLastWriteTime(
    [Parameter(Mandatory=$True)][string] $fileName)
{
    Set-ItemProperty -Path $fileName -Name LastWriteTime -Value (Get-Date) -ErrorAction SilentlyContinue
}

Export-ModuleMember -Function (Write-Output `
    Get-FreeDiskSpaceGB `
    Compress-CacheDirectory `
    Get-TempFileName `
    Update-FileLastWriteTime )

# vim: tabstop=4 shiftwidth=4 expandtab
