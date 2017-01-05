#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#
function PreReqOperations(
    [array] $actionList = @())
{
   foreach ($item in $actionList) {
        foreach ($prereqItem in $item.PreReq) {
            PreRequisiteItem $prereqItem
        }
    }

    Write-Host "Install operations finished"
    Write-Host
}

function PreRequisiteItem(
    [hashtable] $item)
{
    $func = $item["Function"]

    $expr = $func +' $item' 
        
    Write-Verbose "Calling Operation: [$func]"
    Invoke-Expression $expr 
}

function PrereqInfo2013Up5(
    [Parameter(Mandatory = $true)][hashtable] $table
)
{
    FunctionIntro $table

    Write-Host "

We require the installation of Visual Studio 2013 Update 5 to continue. 
Please follow the information here [https://support.microsoft.com/en-us/kb/3021976].
"
}

function PrereqInfoVS15(
    [Parameter(Mandatory = $true)][hashtable] $table
)
{
    FunctionIntro $table
    Write-Host "

Installation of Visual Studio 2015 Update 3 is a pre-requisite before installation 
can continue. Please check [https://www.visualstudio.com/vs/] for more details on
Visual Studion 2015
"
return
}

function PrereqInfoCuda8(
    [Parameter(Mandatory = $true)][hashtable] $table
)
{
    FunctionIntro $table
    Write-Host "

Installation of NVidia Cuda 8.0 is a pre-requisite before installation 
can continue. Please check https://developer.nvidia.com/cuda-downloads] for more 
details on Cuda download. Please also remember to set the environment variable 
CUDA_PATH_V8_0 to the Cuda8 installation location.
"
return
}

function PrereqInfoCuDnn51(
    [Parameter(Mandatory = $true)][hashtable] $table
)
{
    FunctionIntro $table
    Write-Host "

Installation of NVidia CuDNN 4.1 for Cuda 8.0 is a pre-requisite before installation 
can continue. Please check https://developer.nvidia.com/cudn] for more 
details on Cuda download. Please also remember to set the environment variable 
CUDNN_PATH to the CudDnn folder in your installation location.
"
return
}

function StopInstallation(
    [Parameter(Mandatory = $true)][hashtable] $table
)
{
    FunctionIntro $table
    throw "Not all pre-requisites installed, installation terminated."
}


# vim:set expandtab shiftwidth=2 tabstop=2: