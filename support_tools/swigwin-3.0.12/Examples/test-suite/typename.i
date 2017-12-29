%module "typename"

// Tests the typename handling in templates.  

%inline %{
class Foo {
public:
    typedef double Number;
    Number blah() {
        return 2.1828;
    }
};

class Bar {
public:
   typedef int Number;
   Number blah() {
       return 42;
   }
};

template<typename T> typename T::Number twoblah(T &obj) {
   return 2*(obj.blah());
}

Bar::Number spam() { return 3; }

%}

%template(twoFoo) twoblah<Foo>;
%template(twoBar) twoblah<Bar>;


       