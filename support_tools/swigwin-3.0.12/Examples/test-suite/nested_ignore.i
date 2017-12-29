%module nested_ignore
%warnfilter(SWIGWARN_PARSE_NAMED_NESTED_CLASS) B::C::D;

%rename($ignore) B::C;

%inline %{
namespace B {
    class C {
    public:
        struct D {
        };
    };

    class E {
    public:
        typedef C::D D;
    };

    struct F
    {
        const E::D foo(){ return E::D(); }
    };
}
%}
