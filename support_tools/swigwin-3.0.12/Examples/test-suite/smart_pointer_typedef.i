%module smart_pointer_typedef

%inline %{
struct Foo {
   int x;
   int getx() { return x; }
};

typedef Foo *FooPtr;

class Bar {
   Foo *f;
public:
   Bar(Foo *f) : f(f) { }
   FooPtr operator->() {
      return f;
   }
};
%}


