check: all

include ../../Makefile

SUBDIRS := constructor func hierarchy operator hierarchy_operator

.PHONY : all $(SUBDIRS)

all: $(SUBDIRS:%=%-build)
	@for subdir in $(SUBDIRS); do \
		echo Running $$subdir test... ; \
		echo -------------------------------------------------------------------------------- ; \
		cd $$subdir; \
		env LD_LIBRARY_PATH=.:$$LD_LIBRARY_PATH PYTHONPATH=$(srcdir):$$PYTHONPATH $(PYTHON) $(PYSCRIPT); \
		cd ..; \
	done

$(SUBDIRS):
	$(MAKE) -C $@
	@echo Running $$subdir test...
	@echo --------------------------------------------------------------------------------
	cd $@ && env LD_LIBRARY_PATH=.:$$LD_LIBRARY_PATH PYTHONPATH=$(srcdir):$$PYTHONPATH $(PYTHON) $(PYSCRIPT)

%-build:
	$(MAKE) -C $*

%-clean:
	$(MAKE) -s -C $* clean

clean: $(SUBDIRS:%=%-clean)
	rm -f *.pyc
