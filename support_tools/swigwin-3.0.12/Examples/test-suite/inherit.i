// Test that was failing for Perl - the non-member Foo was being called when the member version was intended
%module inherit

%inline %{

const char* Foo(void) {
  return "Non-member Foo";
}

class CBase {
public:
  const char* Foo(void) {
    return "CBase::Foo";
  }
};

class CDerived : public CBase {};

%}
