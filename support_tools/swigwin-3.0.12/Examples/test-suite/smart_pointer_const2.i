%module smart_pointer_const2

%inline %{
struct Foo {
   int x;
   int getx() const { return x; }
   int test() { return x; }
};

class Bar {
   Foo *f;
public:
   Bar(Foo *f) : f(f) { }
   const Foo *operator->() {
      return f;
   }
};
%}


