// Tests some subtle issues of typedef scoping in C++

%module typedef_scope

%inline %{

typedef char * FooType;
class Bar {
public:
     typedef int FooType;
     FooType test1(FooType n, ::FooType data) {
         return n;
     }
     ::FooType test2(FooType n, ::FooType data) {
         return data;
     }
};



class Foo
{
};

typedef Foo FooBar;

class CBaz
{
public:
  typedef FooBar Foo;
};


%}



