// ConfigObjects.h -- objects that the config parser operates on

#pragma once

namespace Microsoft{ namespace MSR { namespace CNTK {

    using namespace std;

    // objects that can print their content

    struct HasToString { virtual wstring ToString() const = 0; };

    // All values that can be used in config files
    //  - are heap objects
    //     - primitives are wrapped
    //     - object pointers are ref-counted shared_ptr, wrapped in ConfigValuePtr
    //  - derive from Object (outside classes get wrapped)
    //
    // This code supports three kinds of value types:
    //  - self-defined classes -> derive from Object, e.g. Expression
    //  - classes defined outside -> wrap in a BoxOf object, e.g. String = BoxOf<wstring>
    //  - C++ primitives like 'double' -> wrap in a Wrapper first then in a BoxOf, e.g. Number = BoxOf<Wrapped<double>> = BoxOfWrapped<double>

    struct Object { virtual ~Object() { } };

    // Wrapped<T> wraps non-class primitive C++ type into a class.
    // (It can also be used for class types, but better use BoxOf<> below directly.)
    template<typename T> class Wrapped
    {
        T value;    // meant to be a primitive type
    public:
        operator const T&() const { return value; }
        operator T&() { return value; }
        Wrapped(T value) : value(value) { }
        T & operator=(const T & newValue) { value = newValue; }
    };
    typedef Wrapped<double> Double;
    typedef Wrapped<bool> Bool;

    // a string (STL wstring, to be precise) that can be help in a ConfigValuePtr
    // TODO: templatize this, call it ConfigObject
    // This can dynamic_cast to wstring.

    // BoxOf<T> wraps a pre-defined type, e.g. std::wstring, to derive from Object.
    // BoxOf<T> can dynamic_cast to T (e.g. BoxOf<wstring> is a wstring).
    template<class C>
    class BoxOf : public Object, public C
    {
    public:
        BoxOf(const C & val) : C(val) { }
        BoxOf(){}
    };
    typedef BoxOf<wstring> String;

}}} // end namespaces
