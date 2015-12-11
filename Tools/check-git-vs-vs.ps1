# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#
# Check files in Git versus files in Visual Studio files (.sln, .vcxproj, .vcxproj.filters)
Param($Baseline)

$ignoreList = Write-Output `
  "Debug|x64" `
  "Release|x64" `
  "Debug|Win32" `
  "Release|Win32" `
  "Debug_CpuOnly|x64" `
  "Release_CpuOnly|x64" `
  "Release_NoOpt|x64"

# TODO ugly
$skip = (Get-Location).Path.Length

function resolvePath($dir, $path) {
    # Note: path may not exist
    $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath("$dir\$path").Substring($skip)
}

function getPathFromMsbuildFile($filePath) {
    $xml = [xml](Get-Content $filePath)
    $ns = New-Object System.Xml.XmlNamespaceManager($xml.NameTable)
    $ns.AddNamespace("ns", "http://schemas.microsoft.com/developer/msbuild/2003")

    $dir = Split-Path $filePath
    $xml.SelectNodes('//ns:*[@Include]', $ns) | ? {
        # Hack to exclude filter nodes from .vcxproj.filters files
        $_.Name -ne 'Filter'
    } | ForEach-Object Include |
    Where-Object { $_ -notin $ignoreList -and $_ -notmatch ';' } |
    ForEach-Object {
        [pscustomobject]@{
            ReferencedFrom = $filePath | % { $_.Fullname.Substring($skip) } # TODO ugly
            Path = $_
            RepoPath = resolvePath $dir $_
        }
    }
}

function getPathsFromSlnFile($filePath) {
    $dir = Split-Path $filePath
    $inSolutionItems = $false
    Get-Content $filePath | ForEach-Object {
        if ($inSolutionItems) {
            if ($_ -match "^`tEndProjectSection$") {
                $inSolutionItems = $false
            } elseif ($_ -match "^`t`t.* = (.*)$") {
                $matches[1]
            }
        } else {
            $inSolutionItems = $_ -match "^`tProjectSection\(SolutionItems\) = preProject$"
        }
    } | ForEach-Object {
        [pscustomobject]@{
            ReferencedFrom = $filePath | % { $_.Fullname.Substring($skip) } # TODO ugly
            Path = $_
            RepoPath = resolvePath $dir $_
        }
    }
}

# TODO should filter out submodule paths
function getPathsFromGit() {
    $dir = Get-Location | ForEach-Object Path
    git ls-tree --name-only -r HEAD | ForEach-Object {
        [pscustomobject]@{
            ReferencedFrom = 'Git HEAD'
                Path = $_
                RepoPath = resolvePath $dir $_
        }
    }
}

$vsPaths = & {
  Get-ChildItem -Recurse -File *.vcxproj.filters, *.vcxproj | ForEach-Object { getPathFromMsbuildFile $_ }
  Get-ChildItem -Recurse -File *.sln | ForEach-Object { getPathsFromSlnFile $_ }
}

$inBaseline = @{}
if ($Baseline) {
  $BaseLine | ForEach-Object { $inBaseline[$_.RepoPath] = 1 }
}

$gitPaths = getPathsFromGit

$inGit = @{}
$gitPaths | ForEach-Object { $inGit[$_.RepoPath] = 1 }

$inVs = @{}
$vsPaths | ForEach-Object { $inVs[$_.RepoPath] = 1 }

$excludeNotInVs = '\.vcxproj(\.filters)?$|\.sln$'

& {
    $vsPaths | ? { -not $inGit[$_.RepoPath] -and -not $inBaseline[$_.RepoPath] } |
        ForEach-Object { $_ | Add-Member -NotePropertyName Status -NotePropertyValue NotInGit -PassThru }
    $gitPaths | ? { -not $inVs[$_.RepoPath] -and $_.RepoPath -notmatch $excludeNotInVs -and -not $inBaseline[$_.RepoPath] } |
        ForEach-Object { $_ | Add-Member -NotePropertyName Status -NotePropertyValue NotInVs -PassThru }
} | Sort-Object RepoPath | Select Status, RepoPath, ReferencedFrom, Path
