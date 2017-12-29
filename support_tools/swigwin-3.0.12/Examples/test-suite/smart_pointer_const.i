%module smart_pointer_const

%inline %{
struct Foo {
   int x;
   int getx() const { return x; }
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


