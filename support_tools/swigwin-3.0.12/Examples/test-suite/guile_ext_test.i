%module guile_ext_test

/* just use the imports_a.h header... for this test we only need a class */
%{
#include "imports_a.h"
%}

%include "imports_a.h"

%{
SCM test_create();
SCM test_is_pointer(SCM val);
%}

%init %{
  scm_c_define_gsubr("test-create", 0, 0, 0, (swig_guile_proc) test_create);
  scm_c_define_gsubr("test-is-pointer", 1, 0, 0, (swig_guile_proc) test_is_pointer);
%}

