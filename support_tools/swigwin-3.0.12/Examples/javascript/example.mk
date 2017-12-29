# Note: as a convention an example must be in a child directory of this.
# These paths are relative to such an example directory

ifneq (, $(ENGINE))
	JSENGINE=$(ENGINE)
else
	JSENGINE=node
endif

ifneq (, $(V8_VERSION))
	JSV8_VERSION=$(V8_VERSION)
else
	JSV8_VERSION=0x031110
endif

EXAMPLES_TOP   = ../..
SWIG_TOP       = ../../..
SWIGEXE        = $(SWIG_TOP)/swig
SWIG_LIB_DIR   = $(SWIG_TOP)/$(TOP_BUILDDIR_TO_TOP_SRCDIR)Lib
TARGET         = example
INTERFACE      = example.i
SWIGOPT        = -$(JSENGINE) -DV8_VERSION=$(JSV8_VERSION)

check: build
	$(MAKE) -f $(EXAMPLES_TOP)/Makefile SRCDIR='$(SRCDIR)' JSENGINE='$(JSENGINE)' TARGET='$(TARGET)' javascript_run

build:
	$(MAKE) -f $(EXAMPLES_TOP)/Makefile SRCDIR='$(SRCDIR)' CXXSRCS='$(SRCS)' \
	SWIG_LIB_DIR='$(SWIG_LIB_DIR)' SWIGEXE='$(SWIGEXE)' \
	SWIGOPT='$(SWIGOPT)' TARGET='$(TARGET)' INTERFACE='$(INTERFACE)' javascript_wrapper_cpp
	$(MAKE) -f $(EXAMPLES_TOP)/Makefile SRCDIR='$(SRCDIR)' CXXSRCS='$(SRCS)' \
	SWIG_LIB_DIR='$(SWIG_LIB_DIR)' SWIGEXE='$(SWIGEXE)' \
	SWIGOPT='$(SWIGOPT)' TARGET='$(TARGET)' INTERFACE='$(INTERFACE)' JSENGINE='$(JSENGINE)' javascript_build_cpp

clean:
	$(MAKE) -f $(EXAMPLES_TOP)/Makefile SRCDIR='$(SRCDIR)' javascript_clean
