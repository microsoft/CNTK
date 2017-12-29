// Test cases for classes that do *NOT* result in smart-pointer wrapping
%module smart_pointer_not

%{
#if defined(__SUNPRO_CC)
#pragma error_messages (off, arrowrtn) /* Questionable return type for ... */
#endif
%}

%inline %{
struct Foo {
   int x;
   int getx() { return x; }
};

class Bar {
   Foo *f;
public:
   Bar(Foo *f) : f(f) { }
   Foo operator->() {
      return *f;
   }
};

class Spam {
   Foo *f;
public:
   Spam(Foo *f) : f(f) { }
   Foo &operator->() {
      return *f;
   }
};

class Grok {
   Foo *f;
public:
   Grok(Foo *f) : f(f) { }
   Foo **operator->() {
      return &f;
   }
};
   
%}


