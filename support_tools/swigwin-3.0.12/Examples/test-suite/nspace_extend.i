// Test the nspace feature and %extend
%module nspace_extend

// nspace feature only supported by these languages
#if defined(SWIGJAVA) || defined(SWIGCSHARP) || defined(SWIGD) || defined(SWIGLUA) || defined(SWIGJAVASCRIPT)

#if defined(SWIGJAVA)
SWIG_JAVABODY_PROXY(public, public, SWIGTYPE)
#endif
%nspace;

%extend Outer::Inner1::Color {
      Color() { return new Outer::Inner1::Color(); }
      virtual ~Color() { delete $self; }
      static Color* create() { return new Outer::Inner1::Color(); }
      Color(const Color& other) { return new Outer::Inner1::Color(other); }

      void colorInstanceMethod(double d) {}
      static void colorStaticMethod(double d) {}
}

%inline %{

namespace Outer {
  namespace Inner1 {
    struct Color {
    };
  }

  namespace Inner2 {
    struct Color {
    };
  }
}
%}

%extend Outer::Inner2::Color {
      Color() { return new Outer::Inner2::Color(); }
      ~Color() { delete $self; }
      static Color* create() { return new Outer::Inner2::Color(); }
      Color(const Color& other) { return new Outer::Inner2::Color(other); }

      void colorInstanceMethod(double d) {}
      static void colorStaticMethod(double d) {}
      void colors(const Inner1::Color& col1a,
                  const Outer::Inner1::Color& col1b,
                  const Color &col2a,
                  const Inner2::Color& col2b,
                  const Outer::Inner2::Color& col2c) {}
}

#else
//#warning nspace feature not yet supported in this target language
#endif

