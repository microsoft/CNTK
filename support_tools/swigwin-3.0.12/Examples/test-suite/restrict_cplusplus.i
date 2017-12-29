%module restrict_cplusplus

%{
// For PHP 5.3 / gcc-4.4
#ifdef restrict
#undef restrict
#endif
struct Foo {
    int restrict;
};
%}

struct Foo {
    int restrict;
};
