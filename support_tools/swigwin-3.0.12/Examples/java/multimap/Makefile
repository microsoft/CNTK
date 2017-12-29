TOP        = ../..
SWIGEXE    = $(TOP)/../swig
SWIG_LIB_DIR = $(TOP)/../$(TOP_BUILDDIR_TO_TOP_SRCDIR)Lib
SRCS       = example.c
TARGET     = example
INTERFACE  = example.i
SWIGOPT    =
JAVASRCS   = *.java

check: build
	$(MAKE) -f $(TOP)/Makefile SRCDIR='$(SRCDIR)' java_run

build:
	$(MAKE) -f $(TOP)/Makefile SRCDIR='$(SRCDIR)' SRCS='$(SRCS)' \
	SWIG_LIB_DIR='$(SWIG_LIB_DIR)' SWIGEXE='$(SWIGEXE)' \
	SWIGOPT='$(SWIGOPT)' TARGET='$(TARGET)' INTERFACE='$(INTERFACE)' java
	$(MAKE) -f $(TOP)/Makefile SRCDIR='$(SRCDIR)' JAVASRCS='$(JAVASRCS)' JAVAFLAGS='$(JAVAFLAGS)' java_compile

clean:
	$(MAKE) -f $(TOP)/Makefile SRCDIR='$(SRCDIR)' java_clean
