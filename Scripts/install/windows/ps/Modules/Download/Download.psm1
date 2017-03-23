#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#

Import-Module Disk -ErrorAction Stop

Set-StrictMode -Version Latest

function DownloadFileWebRequest (
    [Parameter(Mandatory=$True)][string] $SourceFile,
    [Parameter(Mandatory=$True)][string] $OutFile)
{
    try {
        $response = Invoke-WebRequest -Uri $SourceFile -OutFile $OutFile -UserAgent Firefox -TimeoutSec 120 
        return $true
    } 
    catch {
      Write-Verbose "DownloadFileWebRequest failed: $_.Exception.Response.StatusCode.Value__"
      Remove-Item -path $OutFile -ErrorAction SilentlyContinue
      return $false
    }
}

function DownloadFileWebClient(
    [Parameter(Mandatory=$True)][string] $SourceFile,
    [Parameter(Mandatory=$True)][string] $OutFile,
    [int] $timeout = 600)
{
    # $timeout is ignored

    try {
        (New-Object System.Net.WebClient).DownloadFile($SourceFile, $OutFile) 
        return $true
    } 
    catch {
        Write-Verbose "$_.Exception"
        if ($_.Exception.InnerException) {
            Write-Verbose ("Inner exception: {0}" -f $_.Exception.InnerException)
            if ($_.Exception.InnerException.InnerException) {
                Write-Verbose ("Inner inner exception: {0}" -f $_.Exception.InnerException.InnerException)
            }
        }

        Remove-Item -path $OutFile -ErrorAction SilentlyContinue
        return $false
    }
}

function PrepareDownload(
    [string] $targetDir,
    [string] $targetFile)
{
    $tempPrefix = "_inst_"
    if (-not (Test-Path -path $targetDir -PathType Container)) {
        # if we can't create the target directory, we will stop
        New-Item -ItemType Directory -Force -Path $targetDir -ErrorAction Stop
    }

    if (Test-Path -Path $targetFile) {
        Remove-Item -path $OutFile -force -ErrorAction Stop
    }
    remove-Item -path "$targetDir\$tempPrefix*" -ErrorAction SilentlyContinue
    return Get-TempFileName -tempDir $targetDir -filePrefix $tempPrefix 
}

function Copy-FileWebRequest(
    [Parameter(Mandatory=$True)][string] $SourceFile,
    [Parameter(Mandatory=$True)][string] $OutFile,
    [int] $maxtry = 2,
    [int] $tryDelaySeconds = 60)
{
    $targetDir = Split-Path $OutFile
    $workFile = PrepareDownload -targetDir $targetDir -targetFile $outFile

    for ($count=1; $count -le $maxtry; $count +=1) {
        Write-Verbose "Copy-FileWebRequest: Iteration [$count] of [$maxtry]"
        if ($count -gt 1) {
            start-sleep -Seconds $tryDelaySeconds
        }
        if (DownloadFileWebRequest -SourceFile $SourceFile -OutFile $workFile) {
            Rename-Item $workFile $OutFile -Force -ErrorAction Stop | Out-Null
            return $true
        }
    }
    return $false
}

function Copy-FileWebClient(
    [Parameter(Mandatory=$True)][string] $SourceFile,
    [Parameter(Mandatory=$True)][string] $OutFile,
    [int] $timeout = 600,
    [int] $maxtry = 5,
    [int] $tryDelaySeconds = 60)
{
    $targetDir = Split-Path $OutFile
    $workFile = PrepareDownload -targetDir $targetDir -targetFile $outFile

    for ($count=1; $count -le $maxtry; $count +=1) {
        Write-Verbose "Copy-FileWebClient: Iteration [$count] of [$maxtry]"
        if ($count -gt 1) {
            start-sleep -Seconds $tryDelaySeconds
        }
        if (DownloadFileWebClient -SourceFile $SourceFile -OutFile $workFile -timeout $timeout) {
            Rename-Item $workFile $OutFile -Force -ErrorAction Stop | Out-Null
            return $true
        }
    }
    return $false
}
        
Export-ModuleMember -Function (Write-Output `
    Copy-FileWebRequest `
    Copy-FileWebClient )

# vim: tabstop=4 shiftwidth=4 expandtab
