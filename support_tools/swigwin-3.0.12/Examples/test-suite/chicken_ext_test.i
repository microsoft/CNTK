%module chicken_ext_test

/* just use the imports_a.h header... for this test we only need a class */
%{
#include "imports_a.h"
%}

%include "imports_a.h"

%{
void test_create(C_word,C_word,C_word) C_noret;
%}

%init %{
 {
    C_word *space = C_alloc(2 + C_SIZEOF_INTERNED_SYMBOL(11));
    sym = C_intern (&space, 11, "test-create");
    C_mutate ((C_word*)sym+1, (*space=C_CLOSURE_TYPE|1, space[1]=(C_word)test_create, tmp=(C_word)space, space+=2, tmp));
 }
%}

