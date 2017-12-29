%module li_cmalloc

%warnfilter(SWIGWARN_RUBY_WRONG_NAME) sizeof_int;    /* Ruby, wrong constant name */
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) sizeof_double; /* Ruby, wrong constant name */
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) sizeof_intp;   /* Ruby, wrong constant name */

%include <cmalloc.i>

%allocators(int);
%allocators(double);
%allocators(void);
%allocators(int *, intp);
