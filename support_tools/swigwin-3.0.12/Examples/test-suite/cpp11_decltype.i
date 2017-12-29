/* This testcase checks whether SWIG correctly uses the new 'decltype()'
   introduced in C++11.
*/
%module cpp11_decltype

%inline %{
  class A {
  public:
    int i;
    decltype(i) j;

    auto foo( decltype(i) a ) -> decltype(i) {
      if (a==5)
        return 10;
      else
        return 0;
    }
  };
  %}
