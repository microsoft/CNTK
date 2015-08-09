// ConfigObjects.h -- objects that the config parser operates on

#pragma once

namespace Microsoft{ namespace MSR { namespace CNTK {

    using namespace std;

    // All values that can be used in config files
    //  - are heap objects
    //     - primitives are wrapped
    //     - object pointers are ref-counted shared_ptr, wrapped in ConfigValuePtr
    //  - derive from Object (outside classes get wrapped)
    //
    // This code supports three kinds of value types:
    //  - self-defined classes -> derive from Object, e.g. Expression
    //  - classes defined outside -> wrap in a Box object, e.g. String = Box<wstring>
    //  - C++ primitives like 'double' -> wrap in a Wrapper first then in a Box, e.g. Number = Box<Wrapper<double>> = BoxOf<double>

    struct Object { virtual ~Object() { } };

    // Wrapped<T> wraps non-class primitive C++ type into a class.
    // (It can also be used for class types, but better use Box<> below directly.)
    template<typename T> class Wrapped
    {
        T value;    // meant to be a primitive type
    public:
        operator const T&() const { return value; }
        operator T&() { return value; }
        Wrapped(T value) : value(value) { }
        T & operator=(const T & newValue) { value = newValue; }
    };

    // ...no, define the BoxOf without Object; call it BoxOf; then change String to BoxOf

    // a string (STL wstring, to be precise) that can be help in a ConfigValuePtr
    // TODO: templatize this, call it ConfigObject
    // This can dynamic_cast to wstring.

    // Box<T> wrappes a pre-defined type, e.g. std::wstring, to derive from Object.
    // Box<T> can dynamic_cast to T (e.g. Box<wstring> is a wstring).
    template<class C>
    class Box : public Object, public C
    {
    public:
        Box(const C & val) : C(val) { }
        Box(){}
    };
    typedef Box<wstring> String;

    // class to box a primitive C++ type so that it derives from Object
    template<typename T> class BoxOf : public Box<Wrapped<T>>
    {
    public:
        BoxOf(T value) : Box(value) { }
    };

}}} // end namespaces
