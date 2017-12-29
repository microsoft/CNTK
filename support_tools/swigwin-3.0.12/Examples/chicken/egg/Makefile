TOP        = ../..
SWIGEXE    = $(TOP)/../swig
SWIG_LIB_DIR = $(TOP)/../$(TOP_BUILDDIR_TO_TOP_SRCDIR)Lib

check: build
	cd eggs/install && csi ../../test.scm

build: single multi

# This creates an egg which contains only the single module.  Any additional implementation files
# that implement the interface being wrapped should also be added to this egg
single: single_wrap.cxx
	mkdir -p eggs
	tar czf eggs/single.egg single.setup single.scm single_wrap.cxx
	rm -f single.scm single_wrap.cxx

# compile the single module with -nounit
single_wrap.cxx: single.i
	$(SWIGEXE) -chicken -c++ -proxy -nounit single.i

# Now build both mod1 and mod2 into a single egg
multi: mod1_wrap.cxx mod2_wrap.cxx
	mkdir -p eggs
	tar czf eggs/multi.egg multi.setup multi_init.scm mod1.scm mod1_wrap.cxx mod2.scm mod2_wrap.cxx
	rm -f mod1.scm mod1_wrap.cxx mod2.scm mod2_wrap.cxx

mod1_wrap.cxx: mod1.i
	$(SWIGEXE) -chicken -c++ -proxy mod1.i

mod2_wrap.cxx: mod2.i
	$(SWIGEXE) -chicken -c++ -proxy mod2.i

clean:
	rm -rf eggs

# this part is for testing...
setup:
	cd eggs && \
	mkdir -p install && \
	chicken-setup -repository `pwd`/install single.egg && \
	chicken-setup -repository `pwd`/install multi.egg
