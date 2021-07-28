dotnet pack -c Release  /p:Platform=x64 "..\bindings\csharp\CNTKLibraryManagedDll\CNTKLibraryManagedDll.csproj" -o ./
./nuget.exe pack nietras.CNTK.GPU/nietras.CNTK.GPU.nuspec -Properties NoWarn=NU5100,NU5048,NU5128
./nuget.exe pack nietras.CNTK.Deps.Cuda/nietras.CNTK.Deps.Cuda.nuspec -Properties NoWarn=NU5100,NU5048,NU5127
./nuget.exe pack nietras.CNTK.Deps.MKL/nietras.CNTK.Deps.MKL.nuspec -Properties NoWarn=NU5100,NU5048,NU5127
./nuget.exe pack nietras.CNTK.Deps.OpenCV.Zip/nietras.CNTK.Deps.OpenCV.Zip.nuspec -Properties NoWarn=NU5100,NU5048,NU5127
./nuget.exe pack nietras.CNTK.Deps.cuDNN/nietras.CNTK.Deps.cuDNN.nuspec -Properties NoWarn=NU5100,NU5048,NU5127