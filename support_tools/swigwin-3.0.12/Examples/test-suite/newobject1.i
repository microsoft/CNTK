/**
 * The purpose of this test is to confirm that a language module
 * correctly handles the case when C++ class member functions (of both
 * the static and non-static persuasion) have been tagged with the
 * %newobject directive.
 */

%module newobject1

%newobject Foo::makeFoo();
%newobject Foo::makeMore();

%inline %{
class Foo
{
private:
    Foo(const Foo&);
    Foo& operator=(const Foo&);
private:
    static int m_fooCount;
protected:
    Foo() {
        m_fooCount++;
    }
public:
    // Factory function (static)
    static Foo *makeFoo() {
        return new Foo;
    }
    
    // Factory function (regular)
    Foo *makeMore() {
        return new Foo;
    }
    
    // Return the number of instances
    static int fooCount() {
        return m_fooCount;
    }
    
    // Destructor
    ~Foo() {
        m_fooCount--;
    }
};
%}

%{
// Static member initialization (not wrapped)
int Foo::m_fooCount = 0;
%}
