/* This interface file tests whether SWIG handles pointer-reference
   (*&) arguments.

   SWIG 1.3a5 signals a syntax error.
*/

%module pointer_reference

%warnfilter(SWIGWARN_TYPEMAP_SWIGTYPELEAK);                   /* memory leak when setting a ptr/ref variable */

#ifdef SWIGGUILE
/* A silly testing typemap for feeding a doubly indirect integer */
%typemap(in) int *&XYZZY (int temp1, int *temp2) {
   temp1 = scm_to_int($input); temp2 = &temp1; $1 = &temp2;
};
#endif

%inline %{
void foo(int *&XYZZY) {}
%}


// Test pointer reference typemaps shipped with SWIG (add in SWIG 1.3.28 for many languages)
%inline %{
struct Struct {
  int value;
  Struct(int v) : value(v) {}
  static Struct instance;
  static Struct *pInstance;
};

void set(Struct *const& s) {
  Struct::instance = *s;
}
Struct *const& get() {
  return Struct::pInstance;
}
int overloading(int i) {
  return 111;
}
int overloading(Struct *const& s) {
  return 222;
}
%}

%{
Struct Struct::instance = Struct(10);
Struct *Struct::pInstance = &Struct::instance;
%}

