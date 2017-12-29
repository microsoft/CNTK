// Test cases for classes that do *NOT* result in smart-pointer wrapping
%module smart_pointer_multi_typedef

%inline %{
struct Foo {
   int x;
   int getx() { return x; }
};

typedef Foo FooObj;
typedef FooObj *FooPtr;

class Bar {
   Foo *f;
public:
   Bar(Foo *f) : f(f) { }
   FooPtr operator->() {
      return f;
   }
};

typedef Bar BarObj;
typedef Bar &BarRef;

class Spam {
   Bar *b;
public:
   Spam(Bar *b) : b(b) { }
   BarObj operator->() {
      return *b;
   }
};

class Grok {
   Bar *b;
public:
   Grok(Bar *b) : b(b) { }
   BarRef operator->() {
      return *b;
   }
};
   
%}


