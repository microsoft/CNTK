%module template_qualifier

/* Stroustruo, 3rd Ed, C.13.6 */
%inline %{
class X {
public:
    template<int> X *xalloc() { return new X(); }
};

%}

%extend X {
%template(xalloc_int) xalloc<200>;
};

