/* This test case checks whether SWIG correctly parses and ignores the
   keywords "static_assert()" inside the class or struct.
*/
%module cpp11_static_assert

%inline %{
template <typename T>
struct Check1 {
  static_assert(sizeof(int) <= sizeof(T), "not big enough");
};

template <typename T>
class Check2 {
  static_assert(sizeof(int) <= sizeof(T), "not big enough");
};
%}

