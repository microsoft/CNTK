/**
 * The purpose of this test is to confirm that a language module
 * correctly handles the case when a C function has been tagged with the
 * %newobject directive.
 */

%module newobject2

%{
#include <stdlib.h>
%}

%{
/* Global initialization (not wrapped) */
int g_fooCount = 0;
%}

%newobject makeFoo();

%inline %{
/* Struct definition */
typedef struct {
  int dummy;
} Foo;

/* Make one */
Foo *makeFoo() {
    Foo *foo = (Foo *) malloc(sizeof(Foo));
    g_fooCount++;
    return foo;
}

/* Return the number of instances */
int fooCount() {
    return g_fooCount;
}

void do_stuff(Foo *f) {
}
%}

%extend Foo {
    ~Foo() {
        free((void *) $self);
	g_fooCount--;
    }
}
