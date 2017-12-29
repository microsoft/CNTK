%module template_ref_type

%inline %{
class X {
public:
    unsigned _i;
};

template <class T> class Container {
public:
    Container () {}
    bool reset () { return false ;}
};

typedef Container<X> XC;
%}

%template(XC) Container<X>;

%inline %{
class Y {
public:
    Y () {};
    bool find (XC &) { return false; }
};
%}

