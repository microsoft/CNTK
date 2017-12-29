#include <iostream>

#include "example.h"

// Overloaded constructors for class Bar
Bar::Bar() {
    std::cout << "Called Bar::Bar()" << std::endl;
}

Bar::Bar(const Bar&) {
    std::cout << "Called Bar::Bar(const Bar&)" << std::endl;
}

Bar::Bar(double x) {
    std::cout << "Called Bar::Bar(double) with x = " << x << std::endl;
}

Bar::Bar(double x, char *y) {
    std::cout << "Called Bar::Bar(double, char *) with x, y = " << x << ", \"" << y << "\"" << std::endl;
}

Bar::Bar(int x, int y) {
    std::cout << "Called Bar::Bar(int, int) with x, y = " << x << ", " << y << std::endl;
}

Bar::Bar(char *x) {
    std::cout << "Called Bar::Bar(char *) with x = \"" << x << "\"" << std::endl;
}

Bar::Bar(int x) {
    std::cout << "Called Bar::Bar(int) with x = " << x << std::endl;
}

Bar::Bar(long x) {
    std::cout << "Called Bar::Bar(long) with x = " << x << std::endl;
}

Bar::Bar(Bar *x) {
    std::cout << "Called Bar::Bar(Bar *) with x = " << x << std::endl;
}

// Overloaded member functions
void Bar::foo(const Bar& x) {
    std::cout << "Called Bar::foo(const Bar&) with &x = " << &x << std::endl;
}

void Bar::foo(double x) {
    std::cout << "Called Bar::foo(double) with x = " << x << std::endl;
}

void Bar::foo(double x, char *y) {
    std::cout << "Called Bar::foo(double, char *) with x, y = " << x << ", \"" << y << "\"" << std::endl;
}

void Bar::foo(int x, int y) {
    std::cout << "Called Bar::foo(int, int) with x, y = " << x << ", " << y << std::endl;
}

void Bar::foo(char *x) {
    std::cout << "Called Bar::foo(char *) with x = \"" << x << "\"" << std::endl;
}

void Bar::foo(int x) {
    std::cout << "Called Bar::foo(int) with x = " << x << std::endl;
}

void Bar::foo(long x) {
    std::cout << "Called Bar::foo(long) with x = " << x << std::endl;
}

void Bar::foo(Bar *x) {
    std::cout << "Called Bar::foo(Bar *) with x = " << x << std::endl;
}

void Bar::spam(int x, int y, int z) {
    std::cout << "Called Bar::spam(int, int, int) with x, y, z = " << x << ", " << y << ", " << z << std::endl;
}

void Bar::spam(double x, int y, int z) {
    std::cout << "Called Bar::spam(double, int, int) with x, y, z = " << x << ", " << y << ", " << z << std::endl;
}

// Overloaded global methods
void foo(const Bar& x) {
    std::cout << "Called foo(const Bar& x) with &x = " << &x << std::endl;
}

void foo(double x) {
    std::cout << "Called foo(double) with x = " << x << std::endl;
}

void foo(double x, char *y) {
    std::cout << "Called foo(double, char *) with x, y = " << x << ", \"" << y << "\"" << std::endl;
}

void foo(int x, int y) {
    std::cout << "Called foo(int, int) with x, y = " << x << ", " << y << std::endl;
}

void foo(char *x) {
    std::cout << "Called foo(char *) with x = \"" << x << "\"" << std::endl;
}

void foo(int x) {
    std::cout << "Called foo(int) with x = " << x << std::endl;
}

void foo(long x) {
    std::cout << "Called foo(long) with x = " << x << std::endl;
}

void foo(Bar *x) {
    std::cout << "Called foo(Bar *) with x = " << x << std::endl;
}

