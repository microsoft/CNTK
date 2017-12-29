%module smart_pointer_rename

%rename(ftest1) Foo::test(int);
%rename(ftest2) Foo::test(int,int);

%inline %{

class Foo {
public:
    int   test(int) { return 1; }
    int   test(int,int) { return 2; }
};

class Bar {
    Foo *f;
public:
    Bar(Foo *_f) : f(_f) { }
    Foo *operator->() { return f; }
    int  test() { return 3; }
};

%}

	

