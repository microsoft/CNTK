%module namespace_class


%warnfilter(SWIGWARN_PARSE_NAMED_NESTED_CLASS) Ala::Ola;

#ifdef SWIGD
%warnfilter(SWIGWARN_IGNORE_OPERATOR_LT);
#endif

%inline %{
  template<class T> void foobar(T t) {}
  namespace test {
    template<class T> void barfoo(T t) {}
  }  
%}

%template(FooBarInt) ::foobar<int>;
%template(BarFooInt) test::barfoo<int>;


%inline %{
  template <class C>
  struct Bar_T
  {
  };
  
  

  namespace test {
    enum Hello {
      Hi
    };    
    
    struct Test;

    struct Bar  {
      Hello foo(Hello h) {
	return h;
      }
    };

    namespace hola {
      struct Bor;
      struct Foo;
      struct Foobar;
      template <class T> struct BarT {
      };

      template <class T> class FooT;
    }

    template <class T>
    class hola::FooT {
    public:
      Hello foo(Hello h) {
	return h;
      }
      
      T bar(T h) {
	return h;
      }
    };

    namespace hola {
      template <> class FooT<double>;
      template <> class FooT<int>;
    }
    
    template <>
    class hola::FooT<double> {
    public:
      double moo(double h) {
	return h;
      }
    };

    int a;

    struct hola::Foo : Bar {
      Hello bar(Hello h) {
	return h;
      }    
    };
  }
  
  struct test::Test {
    Hello foo(Hello h) {
      return h;
    }
  };

  struct test::hola::Bor {
    Hello foo(Hello h) {
      return h;
    }    
  };

  namespace test {
    struct hola::Foobar : Bar {
      Hello bar(Hello h) {
	return h;
      }    
    };
  }

  template <>
  class test::hola::FooT<int> {
  public:
    int quack(int h) {
      return h;
    }
  };

%}


namespace test
{
  namespace hola {
    %template(FooT_i) FooT<int>;
  }

  %template(FooT_H) hola::FooT<Hello>;
}

%template(FooT_d) ::test::hola::FooT<double>;
%template(BarT_H) test::hola::BarT<test::Hello>;

%inline %{

  namespace hi {
    namespace hello {
      template <class T> struct PooT;
    }

    namespace hello {
      template <class T> struct PooT
      {
      }; 
    }
  }
%}

%template(Poo_i) hi::hello::PooT<int>;

%inline %{

  template <class T> struct BooT {
  };

  namespace test {
    
    typedef ::BooT<Hello> BooT_H;
  }

%}

namespace test {
  
  %template(BooT_H) ::BooT<Hello>;
}
%template(BooT_i) ::BooT<int>;


%inline %{

namespace jafar {
  namespace jmath {
    class EulerT3D {
    public:
      static void hello(){}
      
      template<class VecFrame, class Vec, class VecRes>
      static void toFrame(const VecFrame& frame_, const Vec&v_,const VecRes& vRes){}
      
      template<class T>
      void operator ()(T& x){}

      template<class T>
      void operator < (T& x){}
      
      template<class T>
      operator Bar_T<T> () {}

    };
  }
}
%}

%template(toFrame) jafar::jmath::EulerT3D::toFrame<int,int,int>;
%template(callint) jafar::jmath::EulerT3D::operator()<int>;
%template(lessint) jafar::jmath::EulerT3D::operator < <int>;
%template(callfooi) jafar::jmath::EulerT3D::operator() <test::hola::FooT<int> >;
%template(lessfooi) jafar::jmath::EulerT3D::operator < < test::hola::FooT<int> >;


%inline %{

namespace {
  /* the unnamed namespace is 'private', so, the following
     declarations shouldn't be wrapped */
  class Private1
  {
  };

}

namespace a
{
  namespace 
  {
    class Private2
    {
    };
  }
}
 
%}

%inline %{
  class Ala {
  public : 
    Ala() {}
    class Ola {
    public:
      Ola() {}
      void eek() {}
    };
    
    template <class T>
    static void hi() 
    {
    }
  };
%}


%template(hi) Ala::hi<int>;

%extend jafar::jmath::EulerT3D 
{
  
}

%rename(FLACFile) TagLib::FLAC::File;

%inline {
namespace TagLib
{
  class File {
  public:
    File() {}
  };

  class AudioProperties {
  };

  class AudioPropertiesFile {
  public:
    typedef TagLib::File File;
  };
  
  namespace FLAC
  {
    class File;
    class Properties : public AudioProperties  {
    public:
      Properties(File *) {}
    };

    class PropertiesFile : public AudioPropertiesFile  {
    public:
      PropertiesFile(File * = 0) {}
    };

    namespace bar {
      class PropertiesFree  : public AudioProperties  {
      public:
	PropertiesFree(File *) {}
      };
    }

    class FooFilePrivate : private PropertiesFile  {
    public:
      FooFilePrivate(File *) {}
    };

    class FooFile : public PropertiesFile  {
    public:
      FooFile(File *) {}
    };

    class File {
    public:
      File() {}
    };
  }
}
}
