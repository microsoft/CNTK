// This module tests multiple inheritance, typedef handling, and some
// truly horrible parts of the SWIG type system.   This is only tested
// for Python since not all language modules support multiple-inheritance.
// However, if it works for Python, things should be working for other
// modules.

%module(ruby_minherit="1") minherit

#if defined(SWIGPYTHON) || defined(SWIGRUBY) || defined(SWIGOCAML) || defined(SWIGOCTAVE) || defined(SWIGPERL) || defined(SWIGGO)

%inline %{

class Foo {
private:
    int x;
public:
    Foo() { x = 1; }
    virtual ~Foo() {}
    virtual int  xget() {  return x; };
};
typedef Foo *FooPtr;

FooPtr toFooPtr(Foo *f) { return f; }

class Bar {
private:
    int y;
public:
    Bar() { y = 2; }
    virtual ~Bar() {}
    virtual int yget() { return y; }
};

typedef Bar *BarPtr;
BarPtr toBarPtr(Bar *f) { return f; }

class FooBar : public Foo, public Bar {
private:
    int z;
public:
    FooBar() { z = 3; }
    virtual int zget() { return z; }
};

typedef FooBar *FooBarPtr;
FooBarPtr toFooBarPtr(FooBar *f) { return f; }

class Spam: public FooBar {
private:
    int w;
public:
    Spam() { w = 4; }
    virtual int wget() { return w; }
};

typedef Spam *SpamPtr;
SpamPtr toSpamPtr(Spam *f) { return f; }

int xget(FooPtr f) {
   return f->xget();
}

int yget(BarPtr f) {
   return f->yget();
}

int zget(FooBarPtr f) {
   return f->zget();
}

int wget(SpamPtr f) {
   return f->wget();
}
%}

#endif


// Was causing runtime error in Ruby
%include <std_vector.i>
%template(IntVector) std::vector<int>;

