%module(directors="1") java_director_exception_feature_nspace

%include <std_except.i>

%nspace;  // turn namespace feature on for everything.

#define PACKAGEDOT "java_director_exception_feature_nspacePackage."
#define PACKAGESLASH "java_director_exception_feature_nspacePackage/"
%{
#define PACKAGEDOT "java_director_exception_feature_nspacePackage."
#define PACKAGESLASH "java_director_exception_feature_nspacePackage/"
%}

%{
#if defined(_MSC_VER)
  #pragma warning(disable: 4290) // C++ exception specification ignored except to indicate a function is not __declspec(nothrow)
#endif

#include <string>
%}

%include <std_string.i>

// DEFINE exceptions in header section using std::runtime_error
%{
  #include <exception>
  #include <iostream>

  namespace MyNS {

    struct Exception1 : public std::runtime_error {
      Exception1(const std::string& what):runtime_error(what) {}
    };
    struct Exception2 : public std::runtime_error {
      Exception2(const std::string& what):runtime_error(what) {}
    };
    struct Unexpected : public std::runtime_error {
      Unexpected(const std::string& what):runtime_error(what) {}
    };

  }

%}

// Add an explicit handler for Foo::ping, mapping one java exception back to an 'int'
%feature("director:except") MyNS::Foo::ping {
  jthrowable $error = jenv->ExceptionOccurred();
  if ($error) {
    jenv->ExceptionClear();  // clear java exception since mapping to c++ exception
    if (Swig::ExceptionMatches(jenv,$error,"$packagepath/MyNS/MyJavaException1")) {
      throw 1;
    } else if (Swig::ExceptionMatches(jenv,$error,"$packagepath/MyNS/MyJavaException2")) {
      std::string msg(Swig::JavaExceptionMessage(jenv,$error).message());
      throw MyNS::Exception2(msg);
    } else {
      std::cerr << "Test failed, unexpected exception thrown: " <<
	Swig::JavaExceptionMessage(jenv,$error).message() << std::endl;
      throw std::runtime_error("unexpected exception in Foo::ping");
    }
  }
}

// Use default handler on Foo::pong, with directorthrows typemaps

// directorthrows typemaps for java->c++ conversions
%typemap(directorthrows) MyNS::Exception1,MyNS::Exception2,MyNS::Unexpected  %{
  if (Swig::ExceptionMatches(jenv, $error, "$packagepath/$javaclassname")) {
    std::string msg(Swig::JavaExceptionMessage(jenv,$error).message());
    throw $1_type(msg);
  }
%}

// Override the director:except feature so exception specification is not violated
// (Cannot use built-in default of throw DirectorException)
%feature("director:except") MyNS::Foo::pong %{
  jthrowable $error = jenv->ExceptionOccurred();
  if ($error) {
    jenv->ExceptionClear();
    $directorthrowshandlers
    throw ::MyNS::Unexpected(Swig::JavaExceptionMessage(jenv,$error).message());
  }
%}

// TODO 'throws' typemap emitted by emit_action (emit.cxx) has no way
// to get access to language specific special variables like
// $javaclassname or $packagepath  ("java_director_exception_feature" here)

// throws typemaps for c++->java exception conversions
%typemap(throws,throws=PACKAGEDOT"MyNS.MyJavaException1") MyNS::Exception1 %{
  jclass excpcls = jenv->FindClass(PACKAGESLASH"MyNS/MyJavaException1");
  if (excpcls) {
    jenv->ThrowNew(excpcls, $1.what());
   }
  return $null;
%}

%typemap(throws,throws=PACKAGEDOT"MyNS.MyJavaException1") int %{
  (void)$1;
  jclass excpcls = jenv->FindClass(PACKAGESLASH"MyNS/MyJavaException1");
  if (excpcls) {
    jenv->ThrowNew(excpcls, "Threw some integer");
  }
  return $null;
%}

%typemap(throws,throws=PACKAGEDOT"MyNS.MyJavaException2") MyNS::Exception2 %{
  jclass excpcls = jenv->FindClass(PACKAGESLASH"MyNS/MyJavaException2");
  if (excpcls) {
    jenv->ThrowNew(excpcls, $1.what());
  }
  return $null;
%}


%typemap(throws,throws=PACKAGEDOT"MyNS.MyJavaUnexpected") MyNS::Unexpected %{
  jclass excpcls = jenv->FindClass(PACKAGESLASH"MyNS/MyJavaUnexpected");
  if (excpcls) {
    jenv->ThrowNew(excpcls, $1.what());
  }
  return $null;
%}

// Use generic exception translation approach like python, ruby

%feature("director:except") MyNS::Foo::genericpong {
  jthrowable $error = jenv->ExceptionOccurred();
  if ($error) {
    jenv->ExceptionClear();
    throw Swig::DirectorException(jenv,$error);
  }
}

// %exception with throws attribute.  Need throws attribute for checked exceptions
%feature ("except",throws="Exception")  MyNS::Foo::genericpong %{
%}

%feature ("except",throws="Exception")  MyNS::Bar::genericpong %{
  try { $action }
  catch (Swig::DirectorException & direxcp) {
    direxcp.raiseJavaException(jenv);  // jenv always available in JNI code
    return $null;
  }
%}



%feature("director") Foo;

// Rename exceptions on java side to make translation of exceptions more clear
%rename(MyJavaException1) MyNS::Exception1;
%rename(MyJavaException2) MyNS::Exception2;
%rename(MyJavaUnexpected) MyNS::Unexpected;

%typemap(javabase) ::MyNS::Exception1,::MyNS::Exception2,::MyNS::Unexpected "java.lang.Exception";
%rename(getMessage) what() const;  // Rename all what() methods

namespace MyNS {

  struct Exception1 {
      Exception1(const std::string& what);
      const char * what() const;
  };
  struct Exception2 {
      Exception2(const std::string& what);
      const char * what() const;
  };
  struct Unexpected {
      Unexpected(const std::string& what);
      const char * what() const;
  };

}
// In general it is better to use %catches instead of an exception specification on the method
//   since violating an exception specification calls terminate() preventing catch-all behavior
//   like throwing std::runtime_error.  But an exception specification must be used if the
//   actual interface being wrapped does use them.
%catches(MyNS::Exception1,MyNS::Exception2,MyNS::Unexpected) MyNS::Foo::pong;
%catches(MyNS::Exception1,MyNS::Exception2,MyNS::Unexpected) MyNS::Bar::pong;

%inline %{

namespace MyNS {

class Foo {
public:
  virtual ~Foo() {}
  // ping java implementation throws a java Exception1 or an Exception2 if excp is 1 or 2.
  // pong java implementation throws Exception1,Exception2,Unexpected,NullPointerException for 1,2,3,4
  virtual std::string ping(int excp) throw(int,MyNS::Exception2) = 0;
  virtual std::string pong(int excp) /* throws MyNS::Exception1 MyNS::Exception2 MyNS::Unexpected) */ = 0;
  virtual std::string genericpong(int excp) /* unspecified throws - exception is always DirectorException in C++, translated back to whatever thrown in java */ = 0;
};

// Make a bar from a foo, so a call to Java Bar
// goes Java Bar -> C++ Bar -> C++ Foo -> Java Foo Director

class Bar {
public:
  Bar(Foo* d) { delegate=d; }
  virtual std::string ping(int excp) throw(int,MyNS::Exception2)
  {
    return delegate->ping(excp);
  }

  virtual std::string pong(int excp) /* throws MyNS::Exception1,MyNS::Exception2,MyNS::Unexpected */
  {
    return delegate->pong(excp);
  }

  virtual std::string genericpong(int excp)
  {
    return delegate->genericpong(excp);
  }

private:
  Foo * delegate;
};

}
%}
