// Tests global reference variables:
//  - all non const primitives
//  - const and non const class

%module reference_global_vars

%warnfilter(SWIGWARN_TYPEMAP_SWIGTYPELEAK);      /* memory leak when setting a ptr/ref variable */

%inline %{
class TestClass {
public:
    int num;
    TestClass(int n = 0) : num(n) {}
};
%}

// const class reference variable
%{
const TestClass& global_constTestClass = TestClass(33);
%}
%inline %{
TestClass getconstTC() {
    return global_constTestClass;
}
%}

// Macro to help define similar functions
%define ref(type,name)
%{
static type initial_value_##name;
%}
%inline %{
static type &var_##name = initial_value_##name;
type setref_##name(type &x) {
    var_##name = x;
    return var_##name;
}
type& createref_##name(type x) {
    return *new type(x);
}
type value_##name(type &x) {
    return x;
}
%}
%enddef

// primitive reference variables
ref(bool,               bool);
ref(char,               char);
ref(unsigned char,      unsigned_char);
ref(signed char,        signed_char);
ref(short,              short);
ref(unsigned short,     unsigned_short);
ref(int,                int);
ref(unsigned int,       unsigned_int);
ref(long,               long);
ref(unsigned long,      unsigned_long);
ref(float,              float);
ref(double,             double);
ref(long long,          long_long);
ref(unsigned long long, unsigned_long_long);

// class reference variable
ref(TestClass,          TestClass);

