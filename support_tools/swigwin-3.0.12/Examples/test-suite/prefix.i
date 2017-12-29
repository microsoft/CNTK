// Test that was failing for PHP - the value of the -prefix option was
// ignored
%module prefix

%inline %{

class Foo {
public:
  Foo *get_self() {
    return this;
  }
};

%}
