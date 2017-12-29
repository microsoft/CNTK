%module struct_value

%inline %{

struct Foo {
   int x;
};

struct Bar {
   Foo   a;
   struct Foo b;
};

%}
