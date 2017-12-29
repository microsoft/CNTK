%module voidtest

%inline %{

void globalfunc(void) {
}

class Foo {
public:
   Foo(void) { }
   void memberfunc(void) { }
   static void staticmemberfunc(void) { }
};


void *vfunc1(void *f) { return f; }
void *vfunc2(Foo *f) { return f; }
Foo  *vfunc3(void *f) { return (Foo *) f; }
Foo  *vfunc4(Foo *f) { return f; }

%}
