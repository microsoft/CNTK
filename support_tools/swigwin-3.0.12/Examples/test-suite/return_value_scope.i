%module return_value_scope
%inline %{

namespace Hell {
class Foo {
public:
    Foo(int) { };
};

class Bar {
public:
   typedef Foo fooref;
};

class Spam {
public:
   typedef Bar base;
   typedef base::fooref rettype;
   rettype test() {
       return rettype(1);
   }
};
}
%}




