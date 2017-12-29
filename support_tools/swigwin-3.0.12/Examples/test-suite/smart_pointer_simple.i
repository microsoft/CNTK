%module smart_pointer_simple

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
%}


