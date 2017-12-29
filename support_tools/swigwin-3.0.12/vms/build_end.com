$ set def swig_root:[vms]
$
$ file = f$search("swig_root:[vms.o_alpha]*.obj")
$ newobj = 0 
$ if file .nes. ""
$ then
$    v = f$verify(1)
$    library/replace swig_root:[vms.o_alpha]swig.olb swig_root:[vms.o_alpha]*.obj
$    delete swig_root:[vms.o_alpha]*.obj;*
$    v = f$verify(v)
$    newobj = 1
$ endif
$ file = f$search("swig_root:[vms]swig.exe")
$ if file .eqs. "" .or. newobj
$ then
$    v = f$verify(1)
$    cxxlink/exe=swig_root:[vms]swig.exe -
	/repo=swig_root:[source.modules1_1.cxx_repository] -
        swig_root:[vms.o_alpha]swig.olb/include=swigmain
$    v = f$verify(v)
$ endif
