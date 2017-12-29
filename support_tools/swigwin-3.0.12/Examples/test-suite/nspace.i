// Test the nspace feature
%module nspace

// nspace feature only supported by these languages
#if defined(SWIGJAVA) || defined(SWIGCSHARP) || defined(SWIGD) || defined(SWIGLUA) || defined(SWIGJAVASCRIPT)

#if defined(SWIGJAVA)
SWIG_JAVABODY_PROXY(public, public, SWIGTYPE)
#endif

%nspace;
%nonspace Outer::Inner2::NoNSpacePlease;
%nonspace Outer::Inner2::NoNSpacePlease::ReallyNoNSpaceEnum;

%copyctor;
%ignore Outer::Inner2::Color::Color();

#define CONSTANT100 100

%inline %{

namespace Outer {
  class namespce {
  };
  namespace Inner1 {
    enum Channel { Diffuse, Specular = 0x10, Transmission1 };
    enum { ColorEnumVal1, ColorEnumVal2 = 0x11, ColorEnumVal3 };

    struct Color {
      static Color* create() { return new Color(); }

      enum Channel { Diffuse, Specular = 0x20, Transmission };
      enum { ColorEnumVal1, ColorEnumVal2 = 0x22, ColorEnumVal3 };

      int instanceMemberVariable;
      static int staticMemberVariable;
      static const int staticConstMemberVariable = 222;
      static const Channel staticConstEnumMemberVariable = Transmission;
      void colorInstanceMethod(double d) {}
      static void colorStaticMethod(double d) {}
    }; // Color
    int Color::staticMemberVariable = 0;

    Color namespaceFunction(Color k) { return k; }
    int namespaceVar = 0;
  } // Inner1

  namespace Inner2 {
    enum Channel { Diffuse, Specular = 0x30, Transmission2 };

    struct Color {
      Color() : instanceMemberVariable(0) {}
      static Color* create() { return new Color(); }

      enum Channel { Diffuse, Specular = 0x40, Transmission };
      enum { ColorEnumVal1, ColorEnumVal2 = 0x33, ColorEnumVal3 };

      int instanceMemberVariable;
      static int staticMemberVariable;
      static const int staticConstMemberVariable = 333;
      static const Channel staticConstEnumMemberVariable = Transmission;
      void colorInstanceMethod(double d) {}
      static void colorStaticMethod(double d) {}
      void colors(const Inner1::Color& col1a,
                  const Outer::Inner1::Color& col1b,
                  const Color &col2a,
                  const Inner2::Color& col2b,
                  const Outer::Inner2::Color& col2c) {}
    }; // Color
    int Color::staticMemberVariable = 0;
    class NoNSpacePlease {
      public:
        enum NoNSpaceEnum { NoNspace1 = 1, NoNspace2 = 10 };
        enum ReallyNoNSpaceEnum { ReallyNoNspace1 = 1, ReallyNoNspace2 = 10 };
        static int noNspaceStaticFunc() { return 10; }
    };
  } // Inner2

  // Derived class
  namespace Inner3 {
    struct Blue : Inner2::Color {
      void blueInstanceMethod() {}
    };
  }
  namespace Inner4 {
    struct Blue : Inner2::Color {
      void blueInstanceMethod() {}
    };
  }

  class SomeClass {
  public:
    Inner1::Color::Channel GetInner1ColorChannel() { return Inner1::Color::Transmission; }
    Inner2::Color::Channel GetInner2ColorChannel() { return Inner2::Color::Transmission; }
    Inner1::Channel GetInner1Channel() { return Inner1::Transmission1; }
    Inner2::Channel GetInner2Channel() { return Inner2::Transmission2; }
  }; // SomeClass

} // Outer

namespace Outer {
  struct MyWorldPart2 {};
}

struct GlobalClass {
  void gmethod() {}
};

void test_classes(Outer::SomeClass c, Outer::Inner2::Color cc) {}
%}

#else
//#warning nspace feature not yet supported in this target language
#endif

