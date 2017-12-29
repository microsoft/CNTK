%module li_cpointer

%warnfilter(SWIGWARN_RUBY_WRONG_NAME) doublep; /* Ruby, wrong class name */

%include "cpointer.i"

%pointer_functions(int,intp);
%pointer_class(double,doublep);
%pointer_cast(int, unsigned int, int_to_uint);

