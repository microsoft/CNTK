/*
This module tests whether SWIG sets the '$lextype' variable
correctly.  This variable maintains the literal base name of the
type in the wrapper code - it's therefore usually the same
as '$basetype', but NOT ALWAYS.

In the example below, the typemap definitions are written
for any type of 'Animal', but are parameterized through
preprocessor definitions.  So when wrapping functions which
explicitly reference Giraffes,  the wrapper code can 
behave appropriately for that particular species.

For this to work correctly however, it is critical that
there is a variable which strictly preserves the name
of the type.  '$basetype' doesn't currently do this - 
it sometimes contains 'Giraffe' and sometimes (specifically
the case of arrays) contains 'Animal'.  Since existing
code may rely on that behaviour, we create a new variable
'$lextype' which does what we need.

There is no need for any runtime test here, since if the
code is not functioning properly it will fail to compile.
*/

%module lextype
%{
#include <stdlib.h>
%}

%typemap(in) Animal ()
{
    void *space_needed = malloc(HEIGHT_$1_lextype * WIDTH_$1_lextype);
    $1 = ($1_ltype)space_needed;
}

%typemap(in) Animal[2] ()
{
    void *space_needed = malloc(2 * HEIGHT_$1_lextype * WIDTH_$1_lextype);
    $1 = ($1_ltype)space_needed;
}

%inline %{

#define HEIGHT_Giraffe 100
#define WIDTH_Giraffe 5

typedef void * Animal;
typedef Animal Giraffe;

void eat(Giraffe g) {}
void drink(Giraffe *g) {}
Giraffe mate(Giraffe g[2]) { return g[0]; }

%}
