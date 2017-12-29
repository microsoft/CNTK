TOP        = ../../../../..
SWIGEXE    = $(TOP)/../swig
SWIG_LIB_DIR = $(TOP)/../$(TOP_BUILDDIR_TO_TOP_SRCDIR)Lib
SWIGOPT    =
LIBS       =

build:
	$(MAKE) -f $(TOP)/Makefile SRCDIR='$(SRCDIR)' \
	SWIG_LIB_DIR='$(SWIG_LIB_DIR)' SWIGEXE='$(SWIGEXE)' \
	SWIGOPT='$(SWIGOPT)' LIBS='$(LIBS)' TARGET='bar' INTERFACE='bar.i' python_cpp
	cd pkg3 && $(MAKE) SWIGOPT='$(SWIGOPT)' LIBS='$(LIBS)' build

static:
	$(MAKE) -f $(TOP)/Makefile SRCDIR='$(SRCDIR)' \
	SWIG_LIB_DIR='$(SWIG_LIB_DIR)' SWIGEXE='$(SWIGEXE)' \
	SWIGOPT='$(SWIGOPT)' LIBS='$(LIBS)' TARGET='bar' INTERFACE='bar.i' python_cpp
	cd pkg3 && $(MAKE) SWIGOPT='$(SWIGOPT)' LIBS='$(LIBS)' static

clean:
	$(MAKE) -f $(TOP)/Makefile SRCDIR='$(SRCDIR)' TARGET='bar' python_clean
	cd pkg3 && $(MAKE) clean
