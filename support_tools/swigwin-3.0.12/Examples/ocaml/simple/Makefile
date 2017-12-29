TOP        = ../..
SWIGEXE    = $(TOP)/../swig
SWIG_LIB_DIR = $(TOP)/../$(TOP_BUILDDIR_TO_TOP_SRCDIR)Lib
SRCS       = example.c
TARGET     = example
INTERFACE  = example.i
MLFILE     = example.ml
PROGFILE   = example_prog.ml
OBJS       = example.o

check: build
	$(MAKE) -f $(TOP)/Makefile SRCDIR='$(SRCDIR)' TARGET='$(TARGET)' ocaml_run

build: static

dynamic:
	$(MAKE) -f $(TOP)/Makefile SRCDIR='$(SRCDIR)' SRCS='$(SRCS)' \
	SWIG_LIB_DIR='$(SWIG_LIB_DIR)' SWIGEXE='$(SWIGEXE)' \
	TARGET='$(TARGET)' INTERFACE='$(INTERFACE)' MLFILE='$(MLFILE)' \
	PROGFILE='$(PROGFILE)' OBJS='$(OBJS)' \
	ocaml_dynamic

static:
	$(MAKE) -f $(TOP)/Makefile SRCDIR='$(SRCDIR)' SRCS='$(SRCS)' \
	SWIG_LIB_DIR='$(SWIG_LIB_DIR)' SWIGEXE='$(SWIGEXE)' \
	TARGET='$(TARGET)' INTERFACE='$(INTERFACE)' MLFILE='$(MLFILE)' \
	PROGFILE='$(PROGFILE)' OBJS='$(OBJS)' \
	ocaml_static

toplevel:
	$(MAKE) -f $(TOP)/Makefile SRCDIR='$(SRCDIR)' SRCS='$(SRCS)' \
	SWIG_LIB_DIR='$(SWIG_LIB_DIR)' SWIGEXE='$(SWIGEXE)' \
	TARGET='$(TARGET)' INTERFACE='$(INTERFACE)' MLFILE='$(MLFILE)' \
	PROGFILE='$(PROGFILE)' OBJS='$(OBJS)' \
	ocaml_static_toplevel

clean:
	$(MAKE) -f $(TOP)/Makefile SRCDIR='$(SRCDIR)' MLFILE='$(MLFILE)' ocaml_clean
