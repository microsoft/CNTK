$ set def swig_root:[vms]
$
$ swiglib = "swig_root:[vms.o_alpha]swig.olb
$
$ if (f$search("swig_root:[vms]o_alpha.dir") .eqs. "") then $ -
	create/dir swig_root:[vms.o_alpha]
$
$ copy swigconfig.h [-.source.include]
$ copy swigver.h [-.source.include]
$
$ if (f$search("''swiglib'") .eqs. "") then $ -
          library/create/object 'swiglib'
$
