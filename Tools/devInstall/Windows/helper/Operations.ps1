#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#
function OpAnaconda3411(
    [parameter(Mandatory=$true)][string] $cache,
    [parameter(Mandatory=$true)][string] $AnacondaBasePath)
{
    $targetFolder = Split-Path $AnacondaBasePath -Parent
    $prodSubDir = Split-Path $AnacondaBasePath -Leaf
    $prodName = "Anaconda3-4.1.1"
    $prodFile = "Anaconda3-4.1.1-Windows-x86_64.exe"
    $targetPath = join-path $targetFolder $prodSubDir
    $downloadSource = "https://cntkbuildstorage.blob.core.windows.net/cntk-ci-dependencies/anaconda3/4.1.1/Anaconda3-4.1.1-Windows-x86_64.exe"
    $expectedHash = "B4889513DC574F9D6F96DB089315D69D293F8B17635DA4D2E6EEE118DC105F38"

    @( @{ShortName = "ANA3-411"; Name = $prodName;  VerifyInfo = "Checking for $prodName in $targetPath"; ActionInfo = "Installing $prodName";
         Verification = @( @{Function = "VerifyDirectory"; Path = $targetPath; } );
         Download = @( @{Function = "Download"; Method = "WebRequest"; Source = $downloadSource; Destination = "$cache\$prodFile"; ExpectedHash = $expectedHash } );
         # command line parameters for Anaconda installer: /D=$targetPath must be the last parameter and can not be surrounded by quotes
         Action = @( @{Function = "InstallExe"; Command = "$cache\$prodFile"; Param = "/InstallationType=JustMe /AddToPath=0 /RegisterPython=0 /S /D=$targetPath"; runAs=$false; Message = ".... This will take some time. Please be patient ...." } );
     } )
}

function OpAnacondaEnv(
    [parameter(Mandatory=$true)][string] $AnacondaBasePath,
    [parameter(Mandatory=$true)][string] $repoDir,
    [parameter(Mandatory=$true)][string] $reponame,
    [string] $environmentName = "",
    [parameter(Mandatory=$true)][string] $pyVersion)
{
    $prodName = "Python $pyVersion Environment"
    $targetFolder = Split-Path $AnacondaBasePath -Parent
    $prodSubDir = Split-Path $AnacondaBasePath -Leaf
    $targetPath = Join-Path $targetFolder $prodSubDir
    if ($environmentName) {
        $envName = $environmentName
    }
    else {
        $envName = "cntk-py$pyVersion"
    }
    $envDir = Join-Path envs $envName
    $envVar = "CNTK_PY$($pyVersion)_PATH";
    $envValue = Join-Path $targetPath $envDir

    $ymlDirectory = Join-Path $repoDir $repoName
    $ymlDirectory = Join-Path $ymlDirectory scripts\install\windows
    $ymlFile = Join-Path $ymlDirectory "conda-windows-cntk-py$($pyVersion)-environment.yml"

    @{ ShortName = "PYENV"; Name = $prodName;  VerifyInfo = "Checking for $prodName in $targetPath"; ActionInfo = "Creating $prodName";
      Verification  = @( @{Function = "VerifyRunAlways" } );
      Action = @( @{Function = "InstallYml"; BasePath = $targetPath; Env = $envName; ymlFile= $ymlFile },
                  @{Function = "SetEnvironmentVariable"; EnvVar= $envVar; Content = $envValue } )
     }
}

function OpBoost160VS15(
    [parameter(Mandatory=$true)][string] $cache,
    [parameter(Mandatory=$true)][string] $targetFolder)
{
    $prodName = "Boost 1.60.0"
    $prodFile = "boost_1_60_0-msvc-14.0-64.exe"
    $prodSubDir = "boost_1_60_0-msvc-14.0"
    $targetPath = join-path $targetFolder $prodSubDir
    $downloadSource = "https://cntkbuildstorage.blob.core.windows.net/cntk-ci-dependencies/boost/1.60.0/boost_1_60_0-msvc-14.0-64.exe"
    $envVar = "BOOST_INCLUDE_PATH"
    $envVarLib = "BOOST_LIB_PATH"
    $envContentLib = "$targetPath\lib64-msvc-14.0"
    $expectedHash = "DBC37E8A33895FF67489ABFDC3DA7FF175A4900F2E4124AFF3E359C8F3014D2E"

    @( @{Name = $prodName; ShortName = "BOOST160VS15"; VerifyInfo = "Checking for $prodName in $targetPath"; ActionInfo = "Installing $prodName";
         Verification = @( @{Function = "VerifyDirectory"; Path = "$targetPath" },
                        @{Function = "VerifyEnvironmentAndData"; EnvVar = $envVar; Content = $targetPath },
                        @{Function = "VerifyEnvironmentAndData"; EnvVar = $envVarLib; Content  = $envContentLib } );
        Download = @( @{Function = "Download"; Method = "WebRequest"; UserAgent = "Firefox"; Source = $downloadSource; Destination = "$cache\$prodFile"; ExpectedHash = $expectedHash } );
        Action = @( @{Function = "SetEnvironmentVariable"; EnvVar = $envVar; Content = $targetPath },
                    @{Function = "SetEnvironmentVariable"; EnvVar = $envVarLib; Content  = $envContentLib },
                    @{Function = "InstallExe"; Command = "$cache\$prodFile"; Param = "/dir=`"$targetPath`" /SP- /SILENT /NORESTART"; runAs=$false } );
     } )
}

function OpCMake362(
    [parameter(Mandatory=$true)][string] $cache)
{
    $prodName = "CMake 3.6.2"
    $targetPath = join-path $env:ProgramFiles "cmake\bin"
    $cmakeName = "cmake-3.6.2-win64-x64.msi"
    $downloadSource = "https://cntkbuildstorage.blob.core.windows.net/cntk-ci-dependencies/cmake/3.6.2/cmake-3.6.2-win64-x64.msi"
    $expectedHash = "5EB7C09C23B13742161076401BB2F4EDABD75ECAFE8916C7A401532BC3794DD5"
    
    @( @{ShortName = "CMake362"; Name = $prodName; VerifyInfo = "Checking for $prodName in $targetPath"; ActionInfo = "Installing $prodName";
        Verification = @( @{Function = "VerifyWinProductExists"; Match = "^CMake$"; Version = "3.6.2" }  );
        Download = @( @{Function = "Download"; Method = "WebRequest"; Source = $downloadSource; Destination = "$cache\$cmakeName"; ExpectedHash = $expectedHash } );
        Action = @( @{Function = "InstallMsi"; MsiName =  "$cmakeName" ; MsiDir   = "$cache" }  ,
                    @{Function = "AddToPath"; Dir = "$targetPath" } );
        } )
}

function OpMKLDNN012(
    [parameter(Mandatory=$true)][string] $cache,
    [parameter(Mandatory=$true)][string] $targetFolder)
{
    $prodName = "MKLML and MKL-DNN 0.12 CNTK Prebuild"
    $prodFile = "mklml-mkldnn-0.12.zip"
    $prodSubDir =  "mklml-mkldnn-0.12"

    $targetPath = join-path $targetFolder $prodSubDir
    $envVar = "MKL_PATH"
    $envValue = $targetPath
    $downloadSource = "https://cntkbuildstorage.blob.core.windows.net/cntk-ci-dependencies/mkl-dnn/0.12/mklml-mkldnn-0.12.zip"
    $expectedHash = "13C3D485CF96C216B6460188CE6E120847F1BB16B9F66A4134E56EB5D3A37857"

    @( @{ShortName = "MKLDNN012"; VerifyInfo = "Checking for $prodName in $targetPath"; ActionInfo = "Installing $prodName"; 
         Verification = @( @{Function = "VerifyDirectory"; Path = $targetPath },
                           @{Function = "VerifyEnvironmentAndData"; EnvVar = $envVar; Content = $envValue } );
         Download = @( @{ Function = "Download"; Method = "WebRequest"; Source = $downloadSource; Destination = "$cache\$prodFile"; ExpectedHash = $expectedHash} );
         Action = @( @{Function = "ExtractAllFromZip"; zipFileName = "$cache\$prodFile"; destination = $targetFolder; zipSubTree = $prodSubDir; destinationFolder = $prodSubDir },
                     @{Function = "SetEnvironmentVariable"; EnvVar = $envVar; Content  = $envValue }  );
        } )
}

function OpMSMPI70([parameter(
    Mandatory=$true)][string] $cache)
{
    $remoteFilename = "MSMpiSetup.exe"
    $localFilename = "MSMpiSetup70.exe"
    $downloadSource = "https://cntkbuildstorage.blob.core.windows.net/cntk-ci-dependencies/msmpi/70/$remoteFilename";
    $expectedHash = "7DB377051524EE64D0551735A7A9E9A82402068DC529C0D4CF296E2A616C22AF"

    @( @{Name = "MSMPI Installation"; ShortName = "CNTK"; VerifyInfo = "Checking for installed MSMPI 70"; ActionInfo = "Installing MSMPI 70";
         Verification = @( @{Function = "VerifyWinProductVersion"; Match = "^Microsoft MPI \(\d+\."; Version = "7.0.12437.6"; MatchExact = $false } );
         Download = @( @{Function = "Download"; Method = "WebRequest"; Source = $downloadSource; Destination = "$cache\$localFilename"; ExpectedHash = $expectedHash } );
         Action = @( @{Function = "InstallExe"; Command =  "$cache\$localFilename" ; Param = "/unattend" } )
        } )
}

function OpMSMPI70SDK(
    [parameter(Mandatory=$true)][string] $cache)
{
    $remoteFilename = "msmpisdk.msi"
    $localFilename = "msmpisdk70.msi"
    $downloadSource = "https://cntkbuildstorage.blob.core.windows.net/cntk-ci-dependencies/msmpisdk/70/$remoteFilename";
    $expectedHash = "C28FB6121FE7A5102ED8B011992708039EE878B9F58A34B84AF41AA3622B8F4D"

    @( @{Name = "MSMPI SDK70 Installation"; ShortName = "CNTK"; VerifyInfo = "Checking for installed MSMPI 70 SDK"; ActionInfo = "Installing MSMPI 70 SDK";
         Verification = @( @{Function = "VerifyWinProductVersion"; Match = "^Microsoft MPI SDK \(\d+\."; Version = "7.0.12437.6"; MatchExact = $false } );
         Download = @( @{Function = "Download"; Method = "WebRequest"; Source = $downloadSource; Destination = "$cache\$localFilename"; ExpectedHash = $expectedHash } );
         Action = @( @{Function = "InstallMsi"; MsiName = "$localFilename" ; MsiDir = "$cache" } )
        } )
}

function OpNvidiaCub180(
    [parameter(Mandatory=$true)][string] $cache,
    [parameter(Mandatory=$true)][string] $targetFolder)
{
    $prodName = "NVidia CUB 1.8.0"
    $prodFile = "cub-1.8.0.zip"
    $prodSubDir = "cub-1.8.0"
    $targetPath = join-path $targetFolder $prodSubDir
    $envVar = "CUB_PATH";
    $envValue = $targetPath
    $downloadSource = "https://cntkbuildstorage.blob.core.windows.net/cntk-ci-dependencies/cub/1.8.0/cub-1.8.0.zip"

    @( @{ShortName = "CUB180"; VerifyInfo = "Checking for $prodName in $targetPath"; ActionInfo = "Installing $prodName";
         Verification = @( @{Function = "VerifyDirectory"; Path = "$targetPath" },
                           @{Function = "VerifyEnvironmentAndData"; EnvVar = $envVar; Content = $envValue } );
         Download = @( @{Function = "Download"; Method = "WebRequest"; Source = $downloadSource; Destination = "$cache\$prodFile" } );
         Action = @( @{Function = "ExtractAllFromZip"; zipFileName = "$cache\$prodFile"; destination = "$targetFolder"; destinationFolder = $prodSubDir; zipSubTree= $prodSubDir },
                     @{Function = "SetEnvironmentVariable"; EnvVar= $envVar; Content = $envValue } );
         } )
}

function OpNVidiaCudnn73100(
    [parameter(Mandatory=$true)][string] $cache,
    [parameter(Mandatory=$true)][string] $targetFolder)
{
    $prodName = "NVidia CUDNN 7.3.1 for CUDA 10.0"
    $cudnnWin = "cudnn-10.0-windows10-x64-v7.3.1.20.zip"

    $prodSubDir =  "cudnn-10.0-v7.3.1"
    $targetPath = join-path $targetFolder $prodSubDir
    $envVar = "CUDNN_PATH"
    $envValue = join-path $targetPath "cuda"
    $downloadSource = "https://cntkbuildstorage.blob.core.windows.net/cntk-ci-dependencies/cudnn/7.3.1"

    @( @{ShortName = "CUDNN73100"; VerifyInfo = "Checking for $prodName in $targetPath"; ActionInfo = "Installing $prodName";
         Verification = @( @{Function = "VerifyDirectory"; Path = $targetPath },
                           @{Function = "VerifyDirectory"; Path = $envValue },
                           @{Function = "VerifyEnvironmentAndData"; EnvVar = $envVar; Content = $envValue } );
         Download = @( @{Function = "Download"; Method = "WebRequest"; Source = "$downloadSource/$cudnnWin"; Destination = "$cache\$cudnnWin" } );
         Action = @( @{Function = "ExtractAllFromZip"; zipFileName = "$cache\$cudnnWin"; destination = $targetFolder; destinationFolder = $prodSubDir },
                     @{Function = "SetEnvironmentVariable"; EnvVar = $envVar; Content  = $envValue } );
         })
}

function OpOpenCV31(
    [parameter(Mandatory=$true)][string] $cache,
    [parameter(Mandatory=$true)][string] $targetFolder)
{
    $prodName = "OpenCV-3.1"
    $prodFile = "opencv-3.1.0.exe"
    $prodSubDir = "Opencv3.1.0"
    $targetPath = join-path $targetFolder $prodSubDir
    $envVar = "OPENCV_PATH_V31";
    $envValue = "$targetPath\build"
    $downloadSource = "https://cntkbuildstorage.blob.core.windows.net/cntk-ci-dependencies/opencv/3.1.0/opencv-3.1.0.exe"
    $expectedHash = "0CBB10FAB967111B5B699A44CB224F5D729F8D852D2720CBD5CDB56D8770B7B3"
    $archiveSubTree = "opencv"

    @(  @{ShortName = "OPENCV310"; Name = $prodName; VerifyInfo = "Checking for $prodName in $targetPath"; ActionInfo = "Installing $prodName"; 
          Verification = @( @{Function = "VerifyDirectory"; Path = "$targetPath" },
                            @{Function = "VerifyEnvironmentAndData"; EnvVar = $envVar; Content = $envValue } );
          Download = @( @{ Function = "Download"; Method = "WebRequest"; UserAgent = "Firefox"; Source = $downloadSource; Destination = "$cache\$prodFile"; ExpectedHash = $expectedHash } );
          Action = @( @{Function = "Extract7zipSelfExtractingArchive"; archiveName = "$cache\$prodFile"; destination = "$targetFolder"; destinationFolder = $prodSubDir; archiveSubTree= $archiveSubTree },
                      @{Function = "SetEnvironmentVariable"; EnvVar= $envVar; Content = $envValue } );
         } )
}

function OpProtoBuf310VS17(
    [parameter(Mandatory=$true)][string] $cache,
    [parameter(Mandatory=$true)][string] $targetFolder,
    [parameter(Mandatory=$true)][string] $repoDirectory)
{
    # unzip protobuf source in $protoSourceDir = $targetfolder\src\$prodsubdir
    # create batch file to build protobuf files in $scriptDirectory = $targetFolder\script
    # the script file can be used to create the compiled protobuf libraries in $targetPath = $targetFolder\$prodSubDir

    $prodName = "ProtoBuf 3.1.0 Source"
    $prodSrcSubdir = "protobuf-3.1.0"
    $prodFile = "protobuf310.zip"
    $prodSubDir =  "protobuf-3.1.0-vs17"
    $batchFile = "buildProtoVS17.cmd"

    $protoSourceDir = join-path $targetFolder "src"
    $targetPath = Join-Path $protoSourceDir $prodSrcSubdir
    $scriptDirectory = join-path $targetFolder "script"
    $buildDir = join-path $targetFolder $prodSubDir
    $downloadSource = "https://cntkbuildstorage.blob.core.windows.net/cntk-ci-dependencies/protobuf/3.1.0/protobuf-3.1.0.zip"
    $expectedHash = "C07629F666312E43A4C2415AF77F6442178605A8658D975299C793CB89999212"

    @( @{ShortName = "PROTO310"; VerifyInfo = "Checking for $prodName in $targetPath"; ActionInfo = "Installing $prodName"; 
         Verification = @( @{Function = "VerifyDirectory"; Path = $targetPath } );
         Download = @( @{ Function = "Download"; Method = "WebRequest"; Source = $downloadSource; Destination = "$cache\$prodFile"; ExpectedHash = $expectedHash} );
         Action = @( @{Function = "ExtractAllFromZip"; zipFileName = "$cache\$prodFile"; destination = $protoSourceDir; zipSubTree = $prodSrcSubdir; destinationFolder = $prodSrcSubdir },
                     @{Function = "MakeDirectory"; Path = $scriptDirectory },
                     @{Function = "CreateBuildSimpleBatch"; FileName = "$scriptDirectory\$batchFile"; SourceDir = $targetPath; TargetDir = $buildDir; RepoDirectory = $repoDirectory } );
        } )
}

function OpProtoBuf310VS17Prebuild(
    [parameter(Mandatory=$true)][string] $cache,
    [parameter(Mandatory=$true)][string] $targetFolder)
{
    $prodName = "ProtoBuf 3.1.0 VS17 CNTK Prebuild"
    $prodFile = "protobuf-3.1.0-vs17.zip"
    $prodSubDir =  "protobuf-3.1.0-vs17"

    $targetPath = join-path $targetFolder $prodSubDir
    $envVar = "PROTOBUF_PATH"
    $envValue = $targetPath
    $downloadSource = "https://cntkbuildstorage.blob.core.windows.net/cntk-ci-dependencies/protobuf/3.1.0/protobuf-3.1.0-vs17.zip"
    $expectedHash = "ED0F3215AC60E6AE29B21CBFF53F8876E4CF8B4767FEC525CEF0DA6FDF6A4A73"   

    @( @{ShortName = "PROTO310PRE"; VerifyInfo = "Checking for $prodName in $targetPath"; ActionInfo = "Installing $prodName"; 
         Verification = @( @{Function = "VerifyDirectory"; Path = $targetPath },
                           @{Function = "VerifyEnvironmentAndData"; EnvVar = $envVar; Content = $envValue } );
         Download = @( @{ Function = "Download"; Method = "WebRequest"; Source = $downloadSource; Destination = "$cache\$prodFile"; ExpectedHash = $expectedHash} );
         Action = @( @{Function = "ExtractAllFromZip"; zipFileName = "$cache\$prodFile"; destination = $targetFolder; zipSubTree = $prodSubDir; destinationFolder = $prodSubDir },
                     @{Function = "SetEnvironmentVariable"; EnvVar = $envVar; Content  = $envValue }  );
        } )
}

function OpScanProgram()
{
    @{ ShortName = "SCANPROG"; VerifyInfo = "Scan System for installed programs";
      Verification = @( @{Function = "VerifyScanPrograms" } )
     }
}

function OpSwig3010(
    [parameter(Mandatory=$true)][string] $cache,
    [parameter(Mandatory=$true)][string] $targetFolder)
{
    $prodName = "SWIG 3.0.10"
    $prodFile = "swigwin-3.0.10.zip"
    $prodSubDir =  "swigwin-3.0.10"
    $targetPath = join-path $targetFolder $prodSubDir
    $envVar = "SWIG_PATH"
    $envValue = $targetPath
    $downloadSource = "https://cntkbuildstorage.blob.core.windows.net/cntk-ci-dependencies/swig/3.0.10/swigwin-3.0.10.zip"
    $expectedHash = "68A202EBFC62647495074A190A115B629E84C56D74D3017CCB43E56A4B9B83F6"

    @( @{ShortName = "SWIG3010"; VerifyInfo = "Checking for $prodName in $targetPath"; ActionInfo = "Installing $prodName";  
         Verification = @( @{Function = "VerifyDirectory"; Path = $targetPath },
                           @{Function = "VerifyEnvironmentAndData"; EnvVar = $envVar; Content = $envValue } );
         Download = @( @{ Function = "Download"; Method = "WebRequest"; UserAgent = "Firefox"; Source = $downloadSource; Destination = "$cache\$prodFile"; ExpectedHash = $expectedHash } );
         Action = @( @{Function = "ExtractAllFromZip"; zipFileName = "$cache\$prodFile"; destination = $targetFolder; zipSubTree =$prodSubDir; destinationFolder =$prodSubDir },
                     @{Function = "SetEnvironmentVariable"; EnvVar = $envVar; Content  = $envValue } );
        } )
}

function OpCheckVS2019
{
    @( @{Name = "Verify Installation of VS2019"; ShortName = "PREVS19"; VerifyInfo = "Checking for Visual Studio 2019"; 
                        Verification = @( @{Function = "VerifyWinProductVersion"; Match = "^Visual Studio (Community|Professional|Enterprise) 2019$"; Version = "15.5"; MatchExact = $false} ); 
                        PreReq = @( @{Function = "PrereqInfoVS19" } );
                        Action = @( @{Function = "StopInstallation" } )
                        } )
}

function OpCheckCuda10
{
    $programPath = join-path $env:ProgramFiles "NVIDIA GPU Computing Toolkit\CUDA\v10.2"
    @( @{Name = "Verify Installation of NVidia Cuda 10.2"; ShortName = "PRECUDA102"; VerifyInfo = "Checking for NVidia Cuda 10.2";
         Verification = @( @{Function = "VerifyDirectory"; Path = $programPath },
                           @{Function = "VerifyEnvironmentAndData"; EnvVar = "CUDA_PATH_V10_2"; Content = $programPath } );
         PreReq = @( @{Function = "PrereqInfoCuda10" } );
         Action = @( @{Function = "StopInstallation" } )
        } )
}

function OpZlibVS17(
    [parameter(Mandatory=$true)][string] $cache,
    [parameter(Mandatory=$true)][string] $targetFolder,
    [parameter(Mandatory=$true)][string] $repoDirectory)
{
    # unzip protobuf source in $protoSourceDir = $targetfolder\src\$prodsubdir
    # create batch file to build protobuf files in $scriptDirectory = $targetFolder\script
    # the script file can be used to create the compiled protobuf libraries in $targetPath = $targetFolder\$prodSubDir

    $prodName = "zlib / libzip from source"
    $zlibProdName = "zlib-1.2.8"
    $zlibFilename = "zlib128.zip" 
    # $zlibDownloadSource = "https://netix.dl.sourceforge.net/project/libpng/zlib/1.2.8/zlib128.zip"
    $zlibDownloadSource = "https://cntkbuildstorage.blob.core.windows.net/cntk-ci-dependencies/zlib/1.2.8/zlib128.zip"
    $expectedHashZlib = "879D73D8CD4D155F31C1F04838ECD567D34BEBDA780156F0E82A20721B3973D5"
    
    $libzipProdName = "libzip-1.1.3"
    $libzipFilename = "libzip-1.1.3.tar.gz" 
    $libzipDownloadSource = "https://cntkbuildstorage.blob.core.windows.net/cntk-ci-dependencies/libzip/1.1.3/libzip-1.1.3.tar.gz"
    $downloadeSizeLibzip = "1FAA5A524DD4A12C43B6344E618EDCE1BF8050DFDB9D0F73F3CC826929A002B0"
    
    $prodSubDir =  "zlib-vs17"
    $batchFile = "buildZlibVS17.cmd"

    $sourceCodeDir = join-path $targetFolder "src"
    $scriptDirectory = join-path $targetFolder "script"
    $targetPath = join-path $targetFolder $prodSubDir
    $envVar = "ZLIB_PATH"
    $envValue = $targetPath
    
    @( @{ShortName = "ZLIBVS17"; VerifyInfo = "Checking for $prodName in $sourceCodeDir"; ActionInfo = "Installing $prodName"; 
         Verification = @( @{Function = "VerifyDirectory"; Path = "$sourceCodeDir\$zlibProdName" },
                           @{Function = "VerifyDirectory"; Path = "$sourceCodeDir\$libzipProdName" },
                           @{Function = "VerifyFile"; Path = "$scriptDirectory\$batchFile" } );
         Download = @( @{ Function = "Download"; Source = $zlibDownloadSource; Destination = "$cache\$zlibFilename"; ExpectedHash = $expectedHashZlib }, 
                       @{ Function = "Download"; Source = $libzipDownloadSource; Destination = "$cache\$libzipFilename"; ExpectedHash = $downloadeSizeLibzip } );
         Action = @( @{Function = "ExtractAllFromZip"; zipFileName = "$cache\$zlibFilename"; destination = $sourceCodeDir; zipSubTree =$zlibProdName; destinationFolder =$zlibProdName },
                     @{Function = "ExtractAllFromTarGz"; SourceFile =  "$cache\$libzipFilename"; TargzFileName = "$libzipFilename"; destination = $sourceCodeDir },
                     @{Function = "MakeDirectory"; Path = $scriptDirectory },
                     @{Function = "CreateBuildZlibBatch"; FileName = "$scriptDirectory\$batchFile"; zlibSourceDir = (join-path $sourceCodeDir $zlibProdName); libzipSourceDir = (join-path $sourceCodeDir $libzipProdName); TargetDir = $targetPath; RepoDirectory = $repoDirectory } );
        } )
}

function OpZlibVS17Prebuild(
    [parameter(Mandatory=$true)][string] $cache,
    [parameter(Mandatory=$true)][string] $targetFolder)
{
    $prodName = "ZLib VS17 CNTK Prebuild"
    $prodFile = "zlib-vs17.zip"
    $prodSubDir =  "zlib-vs17"


    $targetPath = join-path $targetFolder $prodSubDir
    $envVar = "ZLIB_PATH"
    $envValue = $targetPath
    $downloadSource = "https://cntkbuildstorage.blob.core.windows.net/cntk-ci-dependencies/zlib/vs17/zlib-vs17.zip"
    $expectedHash = "40A79007EC792756370C35E6C8585C0C5E8750A44BD2F60DB1EA542AAF398A7B"

    @( @{ShortName = "ZLIBPRE"; VerifyInfo = "Checking for $prodName in $targetPath"; ActionInfo = "Installing $prodName"; 
         Verification = @( @{Function = "VerifyDirectory"; Path = $targetPath },
                           @{Function = "VerifyEnvironmentAndData"; EnvVar = $envVar; Content = $envValue } );
         Download = @( @{ Function = "Download"; Method = "WebRequest"; Source = $downloadSource; Destination = "$cache\$prodFile"; ExpectedHash = $expectedHash} );
         Action = @( @{Function = "ExtractAllFromZip"; zipFileName = "$cache\$prodFile"; destination = $targetFolder; zipSubTree = $prodSubDir; destinationFolder = $prodSubDir },
                     @{Function = "SetEnvironmentVariable"; EnvVar = $envVar; Content  = $envValue }  );
        } )
}

# vim:set expandtab shiftwidth=4 tabstop=4: