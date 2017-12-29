TOP        = ../..
SWIGEXE    = $(TOP)/../swig
SWIG_LIB_DIR = $(TOP)/../$(TOP_BUILDDIR_TO_TOP_SRCDIR)Lib
TARGET     = dual
GENCXXSRCS = example2_wrap.cxx
INTERFACE  = dual.i
LUA_INTERP = dual.cpp

# This is a little different to normal as we need to static link two modules and a custom interpreter
# We need the external runtime, then swig examples2, and build the module as normal
check: build
	$(MAKE) -f $(TOP)/Makefile SRCDIR='$(SRCDIR)' TARGET='$(TARGET)' lua_embed_run

build:
	$(MAKE) -f $(TOP)/Makefile SRCDIR='$(SRCDIR)' \
	SWIG_LIB_DIR='$(SWIG_LIB_DIR)' SWIGEXE='$(SWIGEXE)' \
	lua_externalhdr
	$(MAKE) -f $(TOP)/Makefile SRCDIR='$(SRCDIR)' \
	SWIG_LIB_DIR='$(SWIG_LIB_DIR)' SWIGEXE='$(SWIGEXE)' \
	SWIGOPT='-c++' INTERFACE='example2.i' lua_swig_cpp
	$(MAKE) -f $(TOP)/Makefile SRCDIR='$(SRCDIR)' $(SWIGLIB) GENCXXSRCS='$(GENCXXSRCS)' \
	SWIG_LIB_DIR='$(SWIG_LIB_DIR)' SWIGEXE='$(SWIGEXE)' \
	SWIGOPT='$(SWIGOPT)' TARGET='$(TARGET)' INTERFACE='example.i' LUA_INTERP='$(LUA_INTERP)' lua_static_cpp

clean:
	$(MAKE) -f $(TOP)/Makefile SRCDIR='$(SRCDIR)' lua_clean
	rm -f swigluarun.h $(TARGET)
