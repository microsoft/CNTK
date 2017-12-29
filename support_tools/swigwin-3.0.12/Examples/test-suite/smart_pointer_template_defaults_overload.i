%module smart_pointer_template_defaults_overload

// SF Bug #1363
// Problem with method overloading when some methods are added by %extend and others are real methods
// and using template default parameters with smart pointers.

%warnfilter(SWIGWARN_LANG_OVERLOAD_IGNORED) Wrap::operator->;
%warnfilter(SWIGWARN_LANG_OVERLOAD_SHADOW) Container::rubout;

%include <std_string.i>
%include <std_map.i>

%inline %{
template <typename T>
class Wrap {
T *ptr;
public:
  Wrap(T *p) : ptr(p) {}
  T const* operator->(void) const { return ptr; }
  T* operator->(void) { return ptr; }
};
%}

%template(StringDoubleMap) std::map<std::string, double>; // erase is generated okay
%template(WrappedMap) Wrap< std::map<std::string, double> >; // erase wrappers lead to compile error

// Above only affects some languages depending on how std::map is implemented.
// Below is a cutdown language independent demonstration of the bug

%extend Container {
    void rubout(int, int) {}
}

%inline %{
template<typename T, typename X = T> class Container {
public:
    int rubout() { return 0; }
    void rubout(T const &element) {}
    static Container* factory() { return new Container(); }
    static Container* factory(bool b) { return new Container(); }
    static void staticstuff(bool) {}
#ifdef SWIG
  %extend {
    void rubout(bool) {}
  }
#endif
};
%}

%extend Container {
    void rubout(int) {}
}

%template(ContainerInt) Container<double>;
%template(WrapContainerInt) Wrap< Container<double> >;

