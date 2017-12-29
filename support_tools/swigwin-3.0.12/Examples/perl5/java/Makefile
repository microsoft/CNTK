TOP        = ../..
SWIGEXE    = $(TOP)/../swig
SWIG_LIB_DIR = $(TOP)/../$(TOP_BUILDDIR_TO_TOP_SRCDIR)Lib
CXXSRCS    =
TARGET     = example
INTERFACE  = example.i
LIBS       = -lm

check: build
	$(MAKE) -f $(TOP)/Makefile SRCDIR='$(SRCDIR)' perl5_run

build: Example.class Example.h
	$(MAKE) -f $(TOP)/Makefile SRCDIR='$(SRCDIR)' CXXSRCS='$(CXXSRCS)' \
	SWIG_LIB_DIR='$(SWIG_LIB_DIR)' SWIGEXE='$(SWIGEXE)' \
	TARGET='$(TARGET)' INTERFACE='$(INTERFACE)' \
	CXXSHARED="gcj -fpic -shared Example.class" PERL5_CCFLAGS='' PERL5_EXP='' LIBS="-lstdc++" perl5_cpp

clean:
	$(MAKE) -f $(TOP)/Makefile SRCDIR='$(SRCDIR)' perl5_clean
	rm -f *.class Example.h

Example.class Example.h: $(SRCDIR)Example.java
	gcj -d . -fPIC -C -c -g $(SRCDIR)Example.java
	gcjh Example.class
