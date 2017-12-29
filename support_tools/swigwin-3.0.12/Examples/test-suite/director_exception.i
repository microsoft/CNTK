%module(directors="1") director_exception

%warnfilter(SWIGWARN_TYPEMAP_DIRECTOROUT_PTR) return_const_char_star;

%{

#if defined(_MSC_VER)
  #pragma warning(disable: 4290) // C++ exception specification ignored except to indicate a function is not __declspec(nothrow)
#endif

#include <string>


// define dummy director exception classes to prevent spurious errors 
// in target languages that do not support directors.

#ifndef SWIG_DIRECTORS
namespace Swig {
class DirectorException {};
class DirectorMethodException: public Swig::DirectorException {};
}
  #ifndef SWIG_fail
    #define SWIG_fail
  #endif
#endif /* !SWIG_DIRECTORS */

%}

%include "std_string.i"

#ifdef SWIGPHP

%feature("director:except") {
	if ($error == FAILURE) {
		throw Swig::DirectorMethodException();
	}
}

%exception {
	try { $action }
	catch (Swig::DirectorException &) { SWIG_fail; }
}

#endif

#ifdef SWIGPYTHON

%feature("director:except") {
	if ($error != NULL) {
		throw Swig::DirectorMethodException();
	}
}

%exception {
	try { $action }
	catch (Swig::DirectorException &) { SWIG_fail; }
}

#endif

#ifdef SWIGJAVA

// Default for director exception warns about unmapped exceptions now in java
// Suppress warnings for this older test
// %warnfilter(476) Bar;

// Default for java is to throw Swig::DirectorException if no
// direct:except feature.  Since methods below have exception specification
// cannot throw director exception.

// Change back to old 2.0 default behavior

%feature("director:except") {
	jthrowable $error = jenv->ExceptionOccurred();
	if ($error) {
	  // Dont clear exception, still be active when return to java execution
	  // Essentially ignore exception occurred -- old behavior.
	  return $null;
	}
}

#endif

#ifdef SWIGRUBY

%feature("director:except") {
    throw Swig::DirectorMethodException($error);
}

%exception {
  try { $action }
  catch (Swig::DirectorException &e) { rb_exc_raise(e.getError()); }
}

#endif

%feature("director") Foo;

%inline {

class Foo {
public:
  virtual ~Foo() {}
  virtual std::string ping() { return "Foo::ping()"; }
  virtual std::string pong(int val=3) { return "Foo::pong();" + ping(); }
};

Foo *launder(Foo *f) {
  return f;
}

}


%{
  struct Unknown1
  {
  };

  struct Unknown2
  {
  };
%}

%feature("director") Bar;
%feature("director") ReturnAllTypes;

%inline %{
  struct Exception1
  {
  };

  struct Exception2
  {
  };

  class Base 
  {
  public:
    virtual ~Base() throw () {}
  };
  

  class Bar : public Base
  {
  public:
    virtual std::string ping() throw (Exception1, Exception2&) { return "Bar::ping()"; }
    virtual std::string pong() throw (Unknown1, int, Unknown2&) { return "Bar::pong();" + ping(); }
    virtual std::string pang() throw () { return "Bar::pang()"; }
  };
  
  // Class to allow regression testing SWIG/PHP not checking if an exception
  // had been thrown in directorout typemaps.
  class ReturnAllTypes
  {
  public:
    int call_int() { return return_int(); }
    double call_double() { return return_double(); }
    const char * call_const_char_star() { return return_const_char_star(); }
    std::string call_std_string() { return return_std_string(); }
    Bar call_Bar() { return return_Bar(); }

    virtual int return_int() { return 0; }
    virtual double return_double() { return 0.0; }
    virtual const char * return_const_char_star() { return ""; }
    virtual std::string return_std_string() { return std::string(); }
    virtual Bar return_Bar() { return Bar(); }
    virtual ~ReturnAllTypes() {}
  };

#ifdef SWIGPYTHON_BUILTIN
bool is_python_builtin() { return true; }
#else
bool is_python_builtin() { return false; }
#endif
%}
