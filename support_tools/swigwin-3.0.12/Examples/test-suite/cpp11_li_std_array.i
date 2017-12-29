%module cpp11_li_std_array

#if defined(SWIGPYTHON) || defined(SWIGRUBY) || defined(SWIGJAVA) || defined(SWIGCSHARP)

%{
#include <array>
%}

%include <std_array.i>

%template(ArrayInt6) std::array<int, 6>;

%inline %{
std::array<int, 6> arrayOutVal() {
  return { -2, -1, 0, 0, 1, 2 };
}

std::array<int, 6> & arrayOutRef() {
  static std::array<int, 6> a = { -2, -1, 0, 0, 1, 2 };
  return a;
}

const std::array<int, 6> & arrayOutConstRef() {
  static std::array<int, 6> a = { -2, -1, 0, 0, 1, 2 };
  return a;
}

std::array<int, 6> * arrayOutPtr() {
  static std::array<int, 6> a = { -2, -1, 0, 0, 1, 2 };
  return &a;
}

std::array<int, 6> arrayInVal(std::array<int, 6> myarray) {
  std::array<int, 6> a = myarray;
  for (auto& val : a) {
    val *= 10;
  }
  return a;
}

const std::array<int, 6> & arrayInConstRef(const std::array<int, 6> & myarray) {
  static std::array<int, 6> a = myarray;
  for (auto& val : a) {
    val *= 10;
  }
  return a;
}

void arrayInRef(std::array<int, 6> & myarray) {
  for (auto& val : myarray) {
    val *= 10;
  }
}

void arrayInPtr(std::array<int, 6> * myarray) {
  for (auto& val : *myarray) {
    val *= 10;
  }
}
%}

#endif
