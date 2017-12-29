%module typemap_array_qualifiers

%define CLEAR_SWIGTYPE_TYPEMAPS
%typemap(in)
   SWIGTYPE,
   SWIGTYPE *,
   SWIGTYPE *const,
   SWIGTYPE *const&,
   const SWIGTYPE *,
   const SWIGTYPE *const,
   const SWIGTYPE *const&,
   const volatile SWIGTYPE *,
   const volatile SWIGTYPE *const,
   const volatile SWIGTYPE *const&,
   SWIGTYPE [],
   SWIGTYPE [ANY],
   const SWIGTYPE [],
   const SWIGTYPE [ANY],
   const volatile SWIGTYPE [],
   const volatile SWIGTYPE [ANY],
   SWIGTYPE &,
   const SWIGTYPE &,
   const volatile SWIGTYPE &
{
%#error Incorrect typemap for $symname: $type
}
%enddef

%inline %{
  typedef struct {
    int a;
  } SomeType;
  typedef SomeType myarray[3];
  typedef const SomeType myconstarray[4];
  typedef volatile SomeType ** mycrazyarray[5];
  extern "C" {
    typedef volatile SomeType (mycrazyfunc)(SomeType);
    typedef volatile SomeType (*mycrazyfuncptr)(SomeType);
  }
%}

CLEAR_SWIGTYPE_TYPEMAPS;
%typemap(in) SWIGTYPE [ANY] {
$1 = 0;
/* Correct typemap for $symname: $type */
}
%inline %{
  void func1a(myarray x) {}
  void func1b(volatile myarray x) {}
%}

CLEAR_SWIGTYPE_TYPEMAPS;
%typemap(in) const SWIGTYPE [ANY] {
$1 = 0;
/* Correct typemap for $symname: $type */
}
%typemap(in) const volatile SWIGTYPE [ANY] {
$1 = 0;
/* Correct typemap for $symname: $type */
}
%inline %{
  void func2a(const myarray x) {}
  void func2b(const myconstarray x) {}
  void func2c(const volatile myconstarray x) {}
%}

CLEAR_SWIGTYPE_TYPEMAPS;
%typemap(in) volatile SWIGTYPE **const [ANY] {
$1 = 0;
/* Correct typemap for $symname: $type */
}
%typemap(in) volatile SWIGTYPE **const [ANY][ANY] {
$1 = 0;
/* Correct typemap for $symname: $type */
}
%inline %{
  void func3a(const mycrazyarray x, const mycrazyarray y[7]) {}
%}

CLEAR_SWIGTYPE_TYPEMAPS;
%typemap(in) SWIGTYPE (*const) (ANY) {
$1 = 0;
/* Correct typemap for $symname: $type */
}
%inline %{
  void func4a(mycrazyfunc *const x, const mycrazyfuncptr y) {}
%}
