%module template_construct

// Tests templates to make sure an extra <> in a constructor is ok.

%inline %{
template<class T> 
class Foo {
    T y;
public:
    Foo<T>(T x) : y(x) { }
};

%}

%template(Foo_int) Foo<int>;
