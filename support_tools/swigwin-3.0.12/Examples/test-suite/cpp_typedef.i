// This file tests SWIG's tracking of C++ typedef declarations

%module cpp_typedef

%{

class Bar {
public:
};
%}

%inline %{
class Foo {
public:
    typedef Bar SomeBar;
    typedef SomeBar SomeOtherBar;
    SomeOtherBar bar() {
        SomeOtherBar b;
        return b;
    }
    static SomeOtherBar sbar() {
        SomeOtherBar b;
        return b;
    }
};

// Test that the correct types are used for typedef struct declarations
typedef struct {
  int something;
  void m() {}
} UnnamedStruct;

typedef struct NamedStruct {
  int something;
  void m() {}
} TypedefNamedStruct;

typedef TypedefNamedStruct DoubleTypedef;

class Test {
public:
    UnnamedStruct test1(UnnamedStruct a) {return a;};
    struct NamedStruct test2(struct NamedStruct a) {return a;};
    TypedefNamedStruct test3(TypedefNamedStruct a) {return a;};
    DoubleTypedef test4(DoubleTypedef a) {return a;};
};

%}
        

