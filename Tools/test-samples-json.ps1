#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#

[CmdletBinding()]
Param([string]$WikiRepoPath)

$scriptDir = Split-Path $MyInvocation.MyCommand.Definition

function GetGitFilesAndDirs {
  [CmdletBinding()]
  Param(
    [parameter(Mandatory=$true)]
    [ref]
    $FileHash,

    [parameter(Mandatory=$true)]
    [ref]
    $DirHash,

    [parameter(Mandatory=$true)]
    [ValidateScript({Test-Path $_})]
    [string]
    $RepoPath,

    [parameter(Mandatory=$true)]
    [string]
    $Treeish
  )

  if ($FileHash.Value -isnot [hashtable]) {
    throw "-FileHash parameter not a [hashtable] reference"
  }
  if ($DirHash.Value -isnot [hashtable]) {
    throw "-DirHash parameter not a [hashtable] reference"
  }
  if (-not (Get-Command git)) {
    throw "git command not in path"
  }

  $FileHash.Value.Clear()
  $DirHash.Value.Clear()

  $originalLocation = Get-Location
  try {
    Set-Location -ErrorAction Stop $RepoPath

    $outFile = git ls-tree -r $Treeish --full-tree --name-only
    if ($LASTEXITCODE -ne 0) {
      throw "git ls-tree failed"
    }

    $outDir = git ls-tree -r $Treeish -d --full-tree --name-only
    if ($LASTEXITCODE -ne 0) {
      throw "git ls-tree -d failed"
    }

    $outFile | ForEach-Object { $FileHash.Value[$_] = $true }
    $outDir | ForEach-Object { $DirHash.Value[$_] = $true }

  } finally {
    Set-Location $originalLocation
  }
}

function SplitPrefix($Prefix, $String) {
  if ($String.StartsWith($Prefix)) {
    $String.Substring($Prefix.Length)
  }
}

# Note: case-sensitive comparison
$sourceFileHash = New-Object -TypeName System.Collections.Hashtable
$sourceDirHash = New-Object -TypeName System.Collections.Hashtable

GetGitFilesAndDirs -FileHash ([ref]$sourceFileHash) -DirHash ([ref]$sourceDirHash) -RepoPath $scriptDir -Treeish HEAD

$wikiFileHash = New-Object -TypeName System.Collections.Hashtable
$wikiDirHash = New-Object -TypeName System.Collections.Hashtable
if ($WikiRepoPath) {
  GetGitFilesAndDirs -FileHash ([ref]$wikiFileHash) -DirHash ([ref]$wikiDirHash) -RepoPath $WikiRepoPath -Treeish HEAD
}

$jsonFile = 'samples.json'

$jsonPath = Join-Path $scriptDir $jsonFile

$json = Get-Content -ErrorAction Stop $jsonPath | ConvertFrom-JSON -ErrorAction Stop

$json.url | ForEach-Object {
  $uri = [Uri]$_

  if ($uri.IsAbsoluteUri -and $uri.Scheme -ceq 'https' -and $uri.Host -ceq 'github.com') {
    $path = $uri.LocalPath
    if ($rest = SplitPrefix '/Microsoft/CNTK/tree/master/' $path) {
      if (-not $sourceDirHash.ContainsKey($rest)) {
        Write-Error "Cannot find $_ as a directory in Git HEAD"
      }
    } elseif ($rest = SplitPrefix '/Microsoft/CNTK/blob/master/' $path) {
      if (-not $sourceFileHash.ContainsKey($rest)) {
        Write-Error "Cannot find $_ as a file in Git HEAD"
      }
    } elseif ($rest = SplitPrefix '/Microsoft/CNTK/wiki/' $path) {
      if ($WikiRepoPath) {
        if (-not ($wikiFileHash.ContainsKey("$rest.md") -or $wikiDirHash.ContainsKey($rest))) {
          Write-Error "Cannot find $_ in Wiki Git HEAD"
        }
      } else {
        Write-Warning "Cannot validate $_, -WikiRepoPath not specified"
      }
    } else {
      throw "Unsupported URL $_"
    }
  } else {
    Write-Error "URL $_ not pointing to https://github.com/"
  }
}
# vim:set expandtab shiftwidth=2 tabstop=2:
