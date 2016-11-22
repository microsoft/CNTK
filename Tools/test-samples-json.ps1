[CmdletBinding()]
Param()

$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path $MyInvocation.MyCommand.Definition

try {
  Push-Location $scriptDir

  # Note: case-sensitive comparison
  $filesInGitHead = New-Object -TypeName System.Collections.Hashtable
  $dirsInGitHead = New-Object -TypeName System.Collections.Hashtable

  git ls-tree -r HEAD --full-tree --name-only | ForEach-Object {
    $filesInGitHead[$_] = $true
  }
  git ls-tree -r HEAD -d --full-tree --name-only | ForEach-Object {
    $dirsInGitHead[$_] = $true
  }
  
  $jsonFile = 'samples.json'
  
  $jsonPath = Join-Path $scriptDir $jsonFile
  
  $null = Test-Path $jsonPath
  
  $json = Get-Content $jsonPath | ConvertFrom-JSON
  
  # TODO more checks
  
  $json.url | ForEach-Object {
    $uri = [Uri]$_
    #$uri
  
    if (-not $uri.IsAbsoluteUri) {
      throw "URL $_ not absolute"
    }
    if (-not ($uri.Scheme -in 'https', 'http')) {
      throw "URL $_ not http:// or https://"
    }
  
    if ($uri.Host -eq 'github.com') {
      if ($uri.LocalPath.StartsWith('/Microsoft/CNTK/tree/master/')) {
        $path = $uri.LocalPath.Substring(28)
        if (-not $dirsInGitHead.ContainsKey($path)) {
          throw "Cannot find $_ as a directory in Git HEAD"
        }
      } elseif ($uri.LocalPath.StartsWith('/Microsoft/CNTK/tree/blob/')) {
        $path = $uri.LocalPath.Substring(26)
        if (-not $filesInGitHead.ContainsKey($path)) {
          throw "Cannot find $_ as a file in Git HEAD"
        }
      }
    }
  }
} finally {
  Pop-Location
}

# vim:set expandtab shiftwidth=2 tabstop=2:
