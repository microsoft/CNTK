#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#

function DownloadOperations(
    [Parameter(Mandatory = $true)][hashtable[]] $operationList)
{
    Write-Host "Performing download operations"

    foreach ($item in $operationList) {
        foreach ($downloadItem in $item.Download) {
            $params = $downloadItem.Params
            $expr = $downloadItem.Function +' @params' 
        
            Write-Verbose "Calling Operation: [$expr]($params)"
            Invoke-Expression $expr
        }
    }

    Write-Host "Download operations finished"
    Write-Host
}

function Download(
    [Parameter(Mandatory = $true)][string] $source,
    [Parameter(Mandatory = $true)][string] $destination)
{
    $downloadOk = Copy-FileWebRequest -SourceFile $source -OutFile $destination -maxtry 2

    if (-not $downloadOk) {
        throw "Download $SourceFile Failed!"
    }
}

# vim:set expandtab shiftwidth=4 tabstop=4: