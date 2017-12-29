TOP        = ../..
SWIGEXE    = $(TOP)/../swig
SWIG_LIB_DIR = $(TOP)/../$(TOP_BUILDDIR_TO_TOP_SRCDIR)Lib
INTERFACE  = example.i
SRCS       =
CXXSRCS    = example.cxx
TARGET     = overload
INCLUDE    =
SWIGOPT    = -proxy -unhideprimitive
VARIANT    =

# uncomment the following lines to build a static exe
#CHICKEN_MAIN = runme.scm
#VARIANT    = _static

check: build
	$(MAKE) -f $(TOP)/Makefile SRCDIR='$(SRCDIR)' chicken_run

build: $(TARGET)

$(TARGET): $(INTERFACE) $(SRCS)
	$(MAKE) -f $(TOP)/Makefile SRCDIR='$(SRCDIR)' \
	SRCS='$(SRCS)' CXXSRCS='$(CXXSRCS)' CHICKEN_MAIN='$(CHICKEN_MAIN)' \
	SWIG_LIB_DIR='$(SWIG_LIB_DIR)' SWIGEXE='$(SWIGEXE)' \
	INCLUDE='$(INCLUDE)' SWIGOPT='$(SWIGOPT)' TARGET='$(TARGET)' \
	INTERFACE='$(INTERFACE)' CHICKENOPTS='$(CHICKENOPTS)' chicken$(VARIANT)_cpp

clean:
	$(MAKE) -f $(TOP)/Makefile SRCDIR='$(SRCDIR)' chicken_clean
	rm -f example.scm
	rm -f $(TARGET)
