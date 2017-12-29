%module overload_subtype

%inline %{

class Foo {};
class Bar : public Foo {};


int  spam(Foo *f) {
    return 1;
}

int spam(Bar *b) {
    return 2;
}

%}
