/* This file tests a few new contract features.
   First it checks to make sure the constant aggregation macro
   %aggregate_check() works.  This is defined in swig.swg.

   Next, it checks to make sure a simple contract works.
   To support contracts, you need to add a macro to the runtime.
   For Python, it looks like this:

#define SWIG_contract_assert(expr, msg)  if (!(expr)) { PyErr_SetString(PyExc_RuntimeError, (char *) msg #expr ); goto fail; } else

   Note: It is used like this:
   SWIG_contract_assert(x == 1, "Some kind of error message");

   Note: Contracts are still experimental.  The runtime interface may
   change in future versions.   
   */

%module aggregate

%include <exception.i>
%aggregate_check(int, check_direction, UP, DOWN, LEFT, RIGHT)

%contract move(int x) {
require:
   check_direction(x);
}

%inline %{
#define UP    1
#define DOWN  2
#define LEFT  3
#define RIGHT 4

int move(int direction) {
    return direction;
}
%}
