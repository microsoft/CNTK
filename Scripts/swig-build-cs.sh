#Please change SWIG_PATH to the installation directory of SWIG, and CNTKDIR to the repository root of CNTK source.
SWIG_PATH=/root/swig-3.0.10
CNTKDIR=/cntk
SOURCEDIR=$CNTKDIR/Source
CXX=/usr/local/mpi/bin/mpic++
#cd $CNTKDIR

rm -f $SOURCEDIR/../bindings/csharp/CNTKLibraryManagedDll/SwigProxyClasses/*
mkdir -p $CNTKDIR/build-mkl/cpu/release/.build/bindings/csharp/Swig

$SWIG_PATH/swig -c++ -csharp -DMSC_VER -I$SOURCEDIR/CNTKv2LibraryDll/API -I$SOURCEDIR/../bindings/common -namespace CNTK -outdir $SOURCEDIR/../bindings/csharp/CNTKLibraryManagedDll/SwigProxyClasses -dllimport CNTKLibraryCSBinding $SOURCEDIR/../bindings/csharp/Swig/cntk_cs.i

$CXX -c $SOURCEDIR/../bindings/csharp/Swig/cntk_cs_wrap.cxx -o $CNTKDIR/build-mkl/cpu/release/.build/bindings/csharp/Swig/cntk_cs_wrap.o -DSWIG -DSWIGCSHARP -D_POSIX_SOURCE -D_XOPEN_SOURCE=600 -D__USE_XOPEN2K -std=c++11 -DCPUONLY -DUSE_MKL -DNDEBUG -DNO_SYNC -DASGD_PARALLEL_SUPPORT  -DKALDI_DOUBLEPRECISION=0 -DHAVE_POSIX_MEMALIGN -DHAVE_EXECINFO_H=1 -DHAVE_CXXABI_H -DHAVE_ATLAS -DHAVE_OPENFST_GE_10400 -DUSE_ZIP -msse4.1 -mssse3 -std=c++0x -fopenmp -fpermissive -fPIC -Werror -fcheck-new -Wno-error=literal-suffix -g -O4 -I$SOURCEDIR/Common/Include -I$SOURCEDIR/CNTKv2LibraryDll -I$SOURCEDIR/CNTKv2LibraryDll/API -I$SOURCEDIR/CNTKv2LibraryDll/proto -I$SOURCEDIR/Math -I$SOURCEDIR/CNTK -I$SOURCEDIR/ActionsLib -I$SOURCEDIR/ComputationNetworkLib -I$SOURCEDIR/SGDLib -I$SOURCEDIR/SequenceTrainingLib -I$SOURCEDIR/CNTK/BrainScript -I$SOURCEDIR/Readers/ReaderLib -I$SOURCEDIR/PerformanceProfilerDll -I/usr/local/protobuf-3.1.0/include -I/usr/local/CNTKCustomMKL/3/include -I/usr/local/kaldi-c024e8aa/src -I/usr/local/kaldi-c024e8aa/tools/ATLAS/include -I/usr/local/kaldi-c024e8aa/tools/openfst/include -I/usr/local/boost-1.60.0/include -I/usr/local/include -I/usr/local/lib/libzip/include -I/usr/local/opencv-3.1.0/include -I$SOURCEDIR/Multiverso/include -ITests/EndToEndTests/CNTKv2Library/Common -I/usr/local/boost-1.60.0/include -I$SOURCEDIR/Readers/CNTKTextFormatReader -MD -MP -MF $CNTKDIR/build-mkl/cpu/release/.build/bindings/csharp/Swig/cntk_cs_wrap.d

$CXX -rdynamic -shared -L$CNTKDIR/build-mkl/cpu/release/lib -L/usr/local/CNTKCustomMKL/3/x64/parallel -L/usr/local/lib -L/usr/local/opencv-3.1.0/lib -L/usr/local/opencv-3.1.0/release/lib -Wl,-rpath,'$ORIGIN' -Wl,-rpath,/usr/local/CNTKCustomMKL/3/x64/parallel -Wl,-rpath,/usr/local/lib -Wl,-rpath,/usr/local/opencv-3.1.0/lib -Wl,-rpath,/usr/local/opencv-3.1.0/release/lib -o $CNTKDIR/build-mkl/cpu/release/lib/libCNTKLibraryCSBinding.so $CNTKDIR/build-mkl/cpu/release/.build/bindings/csharp/Swig/cntk_cs_wrap.o -lm -lmkl_cntk_p -liomp5 -lpthread -lcntkmath -lcntklibrary-2.0 -lmultiverso /usr/local/protobuf-3.1.0/lib/libprotobuf.a

ls  $SOURCEDIR/../bindings/csharp/CNTKLibraryManagedDll/SwigProxyClasses/
ls -l $CNTKDIR/build-mkl/cpu/release/lib 
