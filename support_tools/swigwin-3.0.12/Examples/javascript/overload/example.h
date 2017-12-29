#include <iostream>

void f() {
  std::cout << "Called f()." << std::endl;
}

void f(int val) {
  std::cout << "Called f(int)." << std::endl;
}
void f(int val1, int val2) {
  std::cout << "Called f(int, int)." << std::endl;
}

void f(const char* s) {
  std::cout << "Called f(const char*)." << std::endl;
}

void f(bool val) {
  std::cout << "Called f(bool)." << std::endl;
}

void f(long val) {
  std::cout << "Called f(long)." << std::endl;
}

void f(double val) {
  std::cout << "Called f(double)." << std::endl;
}
