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
      Remove-Item -Path $OutFile -Force -ErrorAction SilentlyContinue
      return $false
    }
}

function DownloadFileWebClient(
    [Parameter(Mandatory=$True)][string] $SourceFile,
    [Parameter(Mandatory=$True)][string] $OutFile)
{
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

      Remove-Item -Path $OutFile -Force -ErrorAction SilentlyContinue
      return $false
    }
}

function Get-FileFromLocation(
    [Parameter(Mandatory=$True)][string] $SourceFile,
    [Parameter(Mandatory=$True)][string] $OutFile,
    [switch] $WebClient,
    [int] $Maxtry = 5,
    [int] $TryDelaySeconds = 60)
{
    $targetDir = Split-Path $OutFile
    if (-not (Test-Path -Path $targetDir -PathType Container)) {
        # if we can't create the target directory, we will stop
        New-Item -ItemType Directory -Force -Path $targetDir -ErrorAction Stop
    }
    if (Test-Path -Path $OutFile) {
        # Outfile already exists
        Write-Verbose "Get-FileFromLocation: Success. [$OutFile] already exists."
        return $true
    }
    $workFile = Get-TempFileName -filePrefix FromLocation -tempDir $targetDir

    try {
        for ($count = 1; $count -le $Maxtry; $count++) {
            Write-Verbose "Copy-FileWeb: Iteration [$count] of [$maxtry]"
            if ($count -gt 1) {
                Start-Sleep -Seconds $TryDelaySeconds
            }
            if ($WebClient) {
                $result = DownloadFileWebClient -SourceFile $SourceFile -OutFile $workFile
            }
            else {
                $result = DownloadFileWebRequest -SourceFile $SourceFile -OutFile $workFile
            }
            if ($result) {
                Rename-Item $workFile $OutFile -Force -ErrorAction Stop | Out-Null
                return $true
            }
        }
        Write-Verbose "Get-FileFromLocation: Failed"
        return $false
    }
    finally {
        Remove-Item $workfile -Force -ErrorAction SilentlyContinue
    }
}
        
Export-ModuleMember -Function (Write-Output `
    Get-FileFromLocation)

# vim: tabstop=4 shiftwidth=4 expandtab
