%module throw_exception

%{
#if defined(_MSC_VER)
  #pragma warning(disable: 4290) // C++ exception specification ignored except to indicate a function is not __declspec(nothrow)
#endif
%}

%warnfilter(SWIGWARN_RUBY_WRONG_NAME) Namespace::enum1;
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) Namespace::enum2;
#ifdef SWIGPHP
%warnfilter(SWIGWARN_PARSE_KEYWORD) Namespace;
#endif

// Tests SWIG's automatic exception mechanism

%inline %{

class CError {
};

void test_is_Error(CError *r) {}

namespace Namespace {
  typedef CError ErrorTypedef;
  typedef const CError& ErrorRef;
  typedef const CError* ErrorPtr;
  typedef int IntArray[10];
  enum EnumTest { enum1, enum2 };
}
class Foo {
public:
    void test_int() throw(int) {
      throw 37;
    }
    void test_msg() throw(const char *) {
      throw "Dead";
    }
    void test_cls() throw(CError) {
      throw CError();
    }	
    void test_cls_ptr() throw(CError *) {
      static CError StaticError;
      throw &StaticError;
    }	
    void test_cls_ref() throw(CError &) {
      static CError StaticError;
      throw StaticError;
    }	
    void test_cls_td() throw(Namespace::ErrorTypedef) {
      throw CError();
    }	
    void test_cls_ptr_td() throw(Namespace::ErrorPtr) {
      static CError StaticError;
      throw &StaticError;
    }	
    void test_cls_ref_td() throw(Namespace::ErrorRef) {
      static CError StaticError;
      throw StaticError;
    }	
    void test_array() throw(Namespace::IntArray) {
      static Namespace::IntArray array;
      for (int i=0; i<10; i++) {
        array[i] = i;
      }
      throw array;
    }	
    void test_enum() throw(Namespace::EnumTest) {
      throw Namespace::enum2;
    }	
    void test_multi(int x) throw(int, const char *, CError) {
      if (x == 1) throw 37;
      if (x == 2) throw "Dead";
      if (x == 3) throw CError();
    }
};

%}

