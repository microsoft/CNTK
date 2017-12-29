%module template_default_class_parms_typedef

// Based on template_default_class_parms testcase but using typedefs in template

%feature("python:nondynamic");

%inline %{
namespace Space {
  struct SomeType {};
  struct AnotherType {};
  template<typename CC, typename DD = SomeType, typename EE = int> class Bar {
  public:
    typedef CC C;
    typedef DD D;
    typedef EE E;
    C CType;
    D DType;
    E EType;
    // Use typedef with no qualifiers
    Bar(C c, D d, E e) {}
    C method(C c, D d, E e) { return c; }

    // Use typedef with classname qualifiers
    Bar(bool, typename Bar::C c, typename Bar::D d, typename Bar::E e) {}
    typename Bar::C method_1(typename Bar::C c, typename Bar::D d, typename Bar::E e) { return c; }

    // Use typedef with classname and full template parameter qualifiers
    Bar(bool, bool, typename Bar<CC, DD, EE>::C c, typename Bar<CC, DD, EE>::D d, typename Bar<CC, DD, EE>::E e) {}
    typename Bar<CC, DD, EE>::C method_2(typename Bar<CC, DD, EE>::C c, typename Bar<CC, DD, EE>::D d, typename Bar<CC, DD, EE>::E e) { return c; }

    // Use typedef with namespace and classname and full template parameter qualifiers
    Bar(bool, bool, bool, typename Space::Bar<CC, DD, EE>::C c, typename Space::Bar<CC, DD, EE>::D d, typename Space::Bar<CC, DD, EE>::E e) {}
    typename Space::Bar<CC, DD, EE>::C method_3(typename Space::Bar<CC, DD, EE>::C c, typename Space::Bar<CC, DD, EE>::D d, typename Space::Bar<CC, DD, EE>::E e) { return c; }
  };
  template<typename TT = SomeType> class Foo {
  public:
    typedef TT T;
    T TType;

    // Use typedef with no qualifiers
    Foo(T t) {}
    T method(T t) { return t; }

    // Use typedef with classname qualifiers
    Foo(const T &, T t) {}
    typename Foo::T method_A(typename Foo::T t) { return t; }

    // Use typedef with classname and full template parameter qualifiers
    Foo(const typename Foo<TT>::T &, const typename Foo<TT>::T &, typename Foo<TT>::T t) {}
    typename Foo<TT>::T method_B(typename Foo<TT>::T t) { return t; }

    // Use typedef with namespace and classname and full template parameter qualifiers
    Foo(const typename Foo<TT>::T &, const typename Foo<TT>::T &, const typename Foo<TT>::T &, typename Foo<TT>::T t) {}
    typename Foo<TT>::T method_C(typename Foo<TT>::T t) { return t; }
  };
  template<typename T = int> class ATemplate {};

  template<typename T> struct UsesBar {
    void use_A(typename Bar<T>::C, typename Bar<T>::D, typename Bar<T>::E) {}
    void use_B(const typename Bar<T>::C &, const typename Bar<T>::D &, const typename Bar<T>::E &) {}
    void use_C(typename Space::Bar<T>::C, typename Space::Bar<T>::D, typename Space::Bar<T>::E) {}
    void use_D(const typename Space::Bar<T>::C &, const typename Space::Bar<T>::D &, const typename Space::Bar<T>::E &) {}
  };
}
%}

// Use defaults
%template(DefaultBar) Space::Bar<double>;
%template(DefaultFoo) Space::Foo<>;

// Don't use all defaults
%template(BarAnotherTypeBool) Space::Bar<Space::AnotherType, bool>;
%template(FooAnotherType) Space::Foo<Space::AnotherType>;

%template() Space::ATemplate<>;

%template(UsesBarDouble) Space::UsesBar<double>;
