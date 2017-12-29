%module typedef_typedef

// Check C::Bar::Foo resolves to A::Foo in typemap search 

%typemap(in) SWIGTYPE, int "__wrong_in_typemap__will_not_compile__"

%typemap(in) A::Foo {
  $1 = 1234; /* A::Foo in typemap */
}

%inline %{
    struct A
    {
         typedef int Foo;
    };

    struct C
    {
         typedef A Bar;
    };

    struct B
    {
         C::Bar::Foo getValue(C::Bar::Foo intvalue) {
             return intvalue;
         }
    };
%}

/*

  An issue can be the steps resolution.
  1) C::Bar is A. So C::Bar::Foo should be first resolved as A::Foo.
  2) Then A::Foo should be resolved int.
  If the first step is skipped the typemap is not applied.

*/
