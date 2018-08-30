sources = \
					of.c \
					of.h

noinst_LTLIBRARIES = libof.la

libof_la_SOURCES = $(sources)

noinst_PROGRAMS = of_test
of_test_LDADD = libof.la ../libdict/libdict.la
of_test_SOURCES = of_test.c
