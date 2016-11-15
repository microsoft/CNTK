#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#

$operations = @(
    @{Name = "Scan System for installed programs"; ShortName = "SCANPROG"; Info = "Scan System for installed programs";
      Verification = @( @{Function = "VerifyScanPrograms" } )
     },
    @{Name = "Verifying Installation contents"; ShortName = "INSTCONTENT"; Info = "Verifying Installation contents";
      Verification = @( @{Function = "VerifyInstallationContent"; Path = "$cntkRootDir" } )
     },
    @{Name = "Installation VS2012 Runtime"; ShortName = "VS2012"; Info = "Install VS2012 Runtime";
      Verification = @( @{Function = "VerifyWin32ProductExists"; Match = "^Microsoft Visual C\+\+ 2012 x64 Additional Runtime" },
                        @{Function = "VerifyWin32ProductExists"; Match = "^Microsoft Visual C\+\+ 2012 x64 Minimum Runtime" } );
      Action = @( @{Function = "InstallExe"; Command  = "$cntkRootDir\prerequisites\VS2012\vcredist_x64.exe"; Param = "/install /passive /norestart"; Message="Installing VS2012 Runtime...." } )
     },
    @{Name = "Installation VS2013 Runtime"; ShortName = "VS2013"; Info = "Install VS2013 Runtime";
      Verification = @( @{Function = "VerifyWin32ProductExists"; Match = "^Microsoft Visual C\+\+ 2013 x64 Additional Runtime" },
                        @{Function = "VerifyWin32ProductExists"; Match = "^Microsoft Visual C\+\+ 2013 x64 Minimum Runtime" } );
      Action = @( @{Function = "InstallExe"; Command  = "$cntkRootDir\prerequisites\VS2013\vcredist_x64.EXE"; Param = "/install /passive /norestart"; Message="Installing VS2013 Runtime...." } )
     },
    @{Name = "MSMPI Installation"; ShortName = "CNTK"; Info = "Install MSMPI";
      Verification = @( @{Function = "VerifyWin32ProductVersion"; Match = "^Microsoft MPI \(\d+\."; Version = "7.0.12437.6" } );
      Action = @( @{Function = "InstallExe"; Command = "$cntkRootDir\prerequisites\msmpisetup.EXE"; Param = "/unattend"; Message="Installing MSMPI ...." } )
     },
    @{Name = "Anaconda3-4.1.1"; ShortName = "ANA3-411"; Info = "Install Anaconda3-4.1.10";
      Verification = @( @{Function = "VerifyDirectory"; Path = "$AnacondaBasePath"; } );
      Download = @( @{Function = "Download"; Source = "https://repo.continuum.io/archive/Anaconda3-4.1.1-Windows-x86_64.exe"; Destination = "$localCache\Anaconda3-4.1.1-Windows-x86_64.exe" } );
      Action = @( @{Function = "InstallExe"; Command = "$localCache\Anaconda3-4.1.1-Windows-x86_64.exe"; Param = "/InstallationType=JustMe /AddToPath=0 /RegisterPython=0 /S /D=$AnacondaBasePath"; runAs=$false; Message="Installing Anaconda3-4.1.1. This will take several minutes. Please be patient ...."} );
     },
    @{Name = "CNTK Python Environment 3.4"; ShortName = "CNTKPY34"; Info = "Setup CNTK PythonEnvironment 3.4";
      Verification  = @( @{Function = "VerifyDirectory"; Path = "$AnacondaBasePath\envs\cntk-py34"; } );
      Action = @( @{Function = "InstallExe"; Command = "$AnacondaBasePath\Scripts\conda.exe"; Param = "env create --file $MyDir\conda-windows-cntk-py34-environment.yml --prefix $AnacondaBasePath\envs\cntk-py34"; WorkDir = "$AnacondaBasePath\Scripts"; runAs=$false; Message="Setting up CNTK-PY34 environment. Please be patient...." } )
     },
    @{Name = "CNTK WHL Install"; ShortName = "CNTKWHL34"; Info = "Setup/Update CNTK Wheel";
      Verification  = @( @{Function = "VerifyWheelDirectory"; WheelDirectory = "$AnacondaBasePath\envs\cntk-py34\Lib\site-packages\cntk" } );
      Action = @( @{Function = "InstallWheel"; BasePath = "$AnacondaBasePath"; EnvName = "cntk-py34"; WheelDirectory="$AnacondaBasePath\envs\cntk-py34\Lib\site-packages\cntk"; Message="Setup/Update of CNTK Wheel environment. Please be patient...." } )
     },
    @{Name = "Create CNTKPY34 batch file"; ShortName = "BATCH34"; Info = "Create CNTKPY34 batch file";
      Verification  = @( @{Function = "VerifyFile"; Path = "$cntkRootDir\scripts\cntkpy34.bat" } );
      Action = @( @{Function = "CreateBatch"; Filename = "$cntkRootDir\scripts\cntkpy34.bat" } )
     }
)
