// Tests typedef through member pointers

%module typedef_mptr

%{
#if defined(__SUNPRO_CC)
#pragma error_messages (off, badargtype2w) /* Formal argument ... is being passed extern "C" ... */
#pragma error_messages (off, wbadinit) /* Using extern "C" ... to initialize ... */
#endif
%}

#if defined(SWIGPYTHON) || defined(SWIGOCAML)

%inline %{

class Foo {
public:
    int add(int x, int y) {
        return x+y;
    }
    int sub(int x, int y) {
        return x-y;
    }
    int do_op(int x, int y, int (Foo::*op)(int, int)) {
	return (this->*op)(x,y);
    }
};

typedef Foo FooObj;
typedef int Integer;

Integer do_op(Foo *f, Integer x, Integer y, Integer (FooObj::*op)(Integer, Integer)) {
    return f->do_op(x,y,op);
}
%}
#endif

#if defined(SWIGPYTHON) || defined(SWIGOCAML)
%constant int (Foo::*add)(int,int) = &Foo::add;
%constant Integer (FooObj::*sub)(Integer,Integer) = &FooObj::sub;
#endif
