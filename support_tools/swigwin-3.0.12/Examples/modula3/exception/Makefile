TOP        = ../..
SWIGEXE    = $(TOP)/../swig
SWIG_LIB_DIR = $(TOP)/../$(TOP_BUILDDIR_TO_TOP_SRCDIR)Lib
CXXSRCS    = example.cxx
TARGET     = example
INTERFACE  = example.i
SWIGOPT    =
MODULA3SRCS = *.[im]3
MODULA3FLAGS= -o runme

check: build
	$(MAKE) -f $(TOP)/Makefile SRCDIR='$(SRCDIR)' modula3_run

build:
	$(MAKE) -f $(TOP)/Makefile SRCDIR='$(SRCDIR)' CXXSRCS='$(CXXSRCS)' \
	SWIG_LIB_DIR='$(SWIG_LIB_DIR)' SWIGEXE='$(SWIGEXE)' \
	SWIGOPT='$(SWIGOPT)' TARGET='$(TARGET)' INTERFACE='$(INTERFACE)' modula3_cpp
#	$(MAKE) -f $(TOP)/Makefile SRCDIR='$(SRCDIR)' MODULA3SRCS='$(MODULA3SRCS)' MODULA3FLAGS='$(MODULA3FLAGS)' modula3_compile
	m3ppinplace $(MODULA3SRCS)
	mv m3makefile $(MODULA3SRCS) src/
	cm3

clean:
	$(MAKE) -f $(TOP)/Makefile SRCDIR='$(SRCDIR)' modula3_clean
