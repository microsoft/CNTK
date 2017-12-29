// Test cases for classes that do *NOT* result in smart-pointer wrapping
%module smart_pointer_multi

%inline %{
struct Foo {
   int x;
   int getx() { return x; }
};

class Bar {
   Foo *f;
public:
   Bar(Foo *f) : f(f) { }
   Foo *operator->() {
      return f;
   }
};

class Spam {
   Bar *b;
public:
   Spam(Bar *b) : b(b) { }
   Bar operator->() {
      return *b;
   }
};

class Grok {
   Bar *b;
public:
   Grok(Bar *b) : b(b) { }
   Bar &operator->() {
      return *b;
   }
};
   
%}


