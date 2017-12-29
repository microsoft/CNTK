%module protected_rename

/**
 * We should be able to rename Foo::y() to 'x' since the protected
 * member variable of the same name is not wrapped. Thus this test
 * case shouldn't generate any warnings.
 */

%rename(x) Foo::y();

%inline %{
class Foo {
protected:
    int x;
public:
    void y() {}
};

%}
