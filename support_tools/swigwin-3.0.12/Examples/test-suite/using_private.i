%module using_private

%inline %{
class Foo {
public:
     virtual ~Foo() { }
     int x;
     int blah(int xx) { return xx; }
     int defaulted(int i = -1) { return i; }
     virtual void virtualmethod() {}
     virtual void anothervirtual() {}
};

class FooBar : private Foo {
public:
     using Foo::blah;
     using Foo::x;
     using Foo::defaulted;
     using Foo::virtualmethod;
     virtual void anothervirtual() {}
     virtual ~FooBar() {}
};

%}

