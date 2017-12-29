%module add_link

%extend Foo {
Foo *blah() {
   return new Foo();
}
};


%inline %{
class Foo {
public:
  Foo() { };
};

%}




