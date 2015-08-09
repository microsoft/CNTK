// ConfigObjects.h -- objects that the config parser operates on

#pragma once

namespace Microsoft{ namespace MSR { namespace CNTK {

    using namespace std;

    struct Object { virtual ~Object() { } };

    // ...TODO: a ConfigValuePtr should be a shared_ptr to the value directly (such as ComputationNode), while having the base class
    // ...ConfigValues are value structs. E.g. we can copy them to construct a ConfigValuePtrfrom them.

    template<typename T> class wrapped
    {
        T value;
    public:
        operator const T&() const { return value; }
        operator T&() { return value; }
        wrapped(T value) : value(value) { }
        T & operator=(const T & newValue) { value = newValue; }
    };

    // class to box a primitive C++ type so that it derives from Object
    template<typename T> class Wrapped : public Object
    {
    public:
        wrapped<T> value;               // primitive type (e.g. double) or shared_ptr<runtime type>
        Wrapped(T value) : value(value) { }
    };

    // ...no, define the Wrapped without Object; call it Wrapped; then change String to Wrapped

    // a string (STL wstring, to be precise) that can be help in a ConfigValuePtr
    // TODO: templatize this, call it ConfigObject
    // This can dynamic_cast to wstring.
    template<class C>
    class Box : public Object, public C
    {
    public:
        Box(const C & val) : C(val) { }
        Box(){}
    };
    typedef Box<wstring> String;

}}} // end namespaces
