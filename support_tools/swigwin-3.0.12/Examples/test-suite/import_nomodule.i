%module import_nomodule
%{
#include "import_nomodule.h"
%}

// For Python
%warnfilter(SWIGWARN_TYPE_UNDEFINED_CLASS) Bar; // Base class 'Foo' ignored - unknown module name for base. Either import the appropriate module interface file or specify the name of the module in the %import directive. 

%import "import_nomodule.h"

#if !defined(SWIGJAVA) && !defined(SWIGRUBY) && !defined(SWIGCSHARP) && !defined(SWIGD) && !defined(SWIGPYTHON_BUILTIN)

/**
 * The proxy class does not have Bar derived from Foo, yet an instance of Bar
 * can successfully be passed to a proxy function taking a Foo pointer (for some
 * language modules).
 * 
 * This violation of the type system is not possible in Java, C# and D due to
 * static type checking. It's also not (currently) possible in Ruby, but this may
 * be fixable (needs more investigation).
 */

%newobject create_Foo;
%delobject delete_Foo;

%inline %{
Foo *create_Foo() {
   return new Foo();
}

void delete_Foo(Foo *f) {
   delete f;
}

void test1(Foo *f, Integer x) { }

class Bar : public Foo { };

%}

#endif

%inline %{
#ifdef SWIGPYTHON_BUILTIN
bool is_python_builtin() { return true; }
#else
bool is_python_builtin() { return false; }
#endif
%}
