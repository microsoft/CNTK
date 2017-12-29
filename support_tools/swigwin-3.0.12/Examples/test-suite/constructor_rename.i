%module constructor_rename

%{
struct Foo {
    Foo() {}
};
%}

struct Foo {
    %rename(RenamedConstructor) Foo();
    Foo() {}
};
