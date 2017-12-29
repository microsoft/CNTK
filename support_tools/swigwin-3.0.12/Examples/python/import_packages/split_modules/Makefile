TOP        = ../../..
LIBS       =

subdirs = vanilla vanilla_split


check: build
	for s in $(subdirs); do \
		(cd $$s && $(MAKE) check); \
	done

build:
	for s in $(subdirs); do \
		(cd $$s && $(MAKE) SWIGOPT='$(SWIGOPT)' LIBS='$(LIBS)' build); \
	done

static:
	for s in $(subdirs); do \
		(cd $$s && $(MAKE) SWIGOPT='$(SWIGOPT)' LIBS='$(LIBS)' static); \
	done

clean:
	$(MAKE) -f $(TOP)/Makefile SRCDIR='$(SRCDIR)' python_clean
	for s in $(subdirs); do \
		(cd $$s && $(MAKE) clean); \
	done
