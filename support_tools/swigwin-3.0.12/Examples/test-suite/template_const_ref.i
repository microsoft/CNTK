%module template_const_ref
%inline %{
template <class T> class Foo {
public:
	char *bar(const T &obj) {
	    return (char *) "Foo::bar";
        }
};
class Bar { };
%}

%template(Foob) Foo<const Bar *>;
%template(Fooi) Foo<const int *>;

