#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#

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

function Copy-FileWebRequest(
    [Parameter(Mandatory=$True)][string] $SourceFile,
    [Parameter(Mandatory=$True)][string] $OutFile,
    [int] $maxtry = 2,
    [int] $tryDelaySeconds = 60)
{
    $targetDir = Split-Path $OutFile
    if (-not (Test-Path -path $targetDir -PathType Container)) {
        # if we can't create the target directory, we will stop
        New-Item -ItemType Directory -Force -Path $targetDir -ErrorAction Stop
    }
    for ($count=1; $count -le $maxtry; $count +=1) {
        Write-Verbose "Copy-FileWebRequest: Iteration [$count] of [$maxtry]"
        if ($count -gt 1) {
            start-sleep -Seconds $tryDelaySeconds
        }
        if (Test-Path -Path $OutFile) {
            Remove-Item -path $OutFile -ErrorAction Stop
        }  
        if (DownloadFileWebRequest -SourceFile $SourceFile -OutFile $OutFile) {
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
    if (-not (Test-Path -path $targetDir -PathType Container)) {
        # if we can't create the target directory, we will stop
        New-Item -ItemType Directory -Force -Path $targetDir -ErrorAction Stop
    }
    for ($count=1; $count -le $maxtry; $count +=1) {
        Write-Verbose "Copy-FileWebClient: Iteration [$count] of [$maxtry]"
        if ($count -gt 1) {
            start-sleep -Seconds $tryDelaySeconds
        }
        if (Test-Path -Path $OutFile) {
            Remove-Item -path $OutFile -ErrorAction Stop
        }  
        if (DownloadFileWebClient -SourceFile $SourceFile -OutFile $OutFile -timeout $timeout) {
            return $true
        }
    }
    return $false
}
        
Export-ModuleMember -Function (Write-Output `
    Copy-FileWebRequest `
    Copy-FileWebClient )

# vim: tabstop=4 shiftwidth=4 expandtab
