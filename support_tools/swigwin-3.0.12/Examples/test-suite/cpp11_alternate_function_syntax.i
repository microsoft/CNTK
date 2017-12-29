/* This testcase checks whether SWIG correctly uses the new alternate functions
   declarations and definitions introduced in C++11. */
%module cpp11_alternate_function_syntax

%inline %{
struct SomeStruct {
  int addNormal(int x, int y);
  auto addAlternate(int x, int y) -> int;
};
 
auto SomeStruct::addAlternate(int x, int y) -> int {
  return x + y;
}

int SomeStruct::addNormal(int x, int y) {
  return x + y;
}
%}
