%module cpp11_type_traits

// The example in the CPlusPlus11.html documentation.
// This doesn't really directly test functionality in type_traits as it doesn't provide
// much for use by target languages, rather it tests usage of it.

%inline %{
#include <type_traits>

// First way of operating.
template< bool B > struct algorithm {
  template< class T1, class T2 > static int do_it(T1 &, T2 &)  { /*...*/ return 1; }
};

// Second way of operating.
template<> struct algorithm<true> {
  template< class T1, class T2 > static int do_it(T1, T2)  { /*...*/ return 2; }
};

// Instantiating 'elaborate' will automatically instantiate the correct way to operate, depending on the types used.
template< class T1, class T2 > int elaborate(T1 A, T2 B) {
  // Use the second way only if 'T1' is an integer and if 'T2' is
  // a floating point, otherwise use the first way.
  return algorithm< std::is_integral<T1>::value && std::is_floating_point<T2>::value >::do_it(A, B);
}
%}

%template(Elaborate) elaborate<int, int>;
%template(Elaborate) elaborate<int, double>;
