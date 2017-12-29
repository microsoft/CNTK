// Test to check the exception classes in the throws attribute of the typemaps and except feature is working

%module java_throws

// Exceptions are chosen at random but are ones which have to have a try catch block to compile
%typemap(in, throws="	 ClassNotFoundException") int num { 
    $1 = (int)$input;
}
%typemap(freearg, throws="InstantiationException  ") int num "/*not written*/"
%typemap(argout, throws="CloneNotSupportedException		") int num "/*not written*/"
%typemap(check, throws="NoSuchFieldException") int num {
    if ($input == 10) {
        jenv->ExceptionClear();
        jclass excep = jenv->FindClass("java/lang/NoSuchFieldException");
        if (excep)
            jenv->ThrowNew(excep, "Value of 10 not acceptable");
        return $null;
    }
}

// Duplicate exceptions should be removed from the generated throws clause
%typemap(out, throws="IllegalAccessException, NoSuchFieldException,   CloneNotSupportedException    ") short { 
    $result = (jshort)$1; 
}

%inline %{
short full_of_exceptions(int num) {
    return 0;
}
%}


%typemap(throws, throws="IllegalAccessException") int {
    (void)$1;
    jclass excep = jenv->FindClass("java/lang/IllegalAccessException");
    if (excep) {
        jenv->ThrowNew(excep, "Test exception");
    }
    return $null;
}
%inline %{
#if defined(_MSC_VER)
  #pragma warning(disable: 4290) // C++ exception specification ignored except to indicate a function is not __declspec(nothrow)
#endif
bool throw_spec_function(int value) throw (int) { throw (int)0; }
#if defined(_MSC_VER)
  #pragma warning(default: 4290) // C++ exception specification ignored except to indicate a function is not __declspec(nothrow)
#endif
%}

%catches(int) catches_function(int value);
%inline %{
bool catches_function(int value) { throw (int)0; }
%}

// Check newfree typemap throws attribute
%newobject makeTestClass;
%typemap(newfree, throws="NoSuchMethodException") TestClass* "/*not written*/"
%inline %{
class TestClass {
    int x;
public:
    TestClass(int xx) : x(xx) {}
};
TestClass* makeTestClass() { return new TestClass(1000); }
%}


// javain typemap throws attribute
// Will only compile if the fileFunction has a java.io.IOException throws clause as getCanonicalPath() throws this exception
%typemap(jstype) char* someFileArgument "java.io.File"
%typemap(javain, throws="java.io.IOException") char* someFileArgument "$javainput.getCanonicalPath()"

%inline %{
void fileFunction(char* someFileArgument) {}
%}


// javout typemap throws attribute
%typemap(javaout, throws="java.io.IOException") int {
    int returnValue=$jnicall;
    if (returnValue==0) throw new java.io.IOException("some IOException");
    return returnValue;
  }

%inline %{
int ioTest() { return 0; }
%}

// except feature (%javaexception) specifying a checked exception class for the throws clause
%typemap(javabase) MyException "Throwable";
%typemap(javacode) MyException %{
  public static final long serialVersionUID = 0x52151000; // Suppress ecj warning
%}
%inline %{
    struct MyException {
        MyException(const char *msg) {}
    };
%}

%define JAVAEXCEPTION(METHOD)
%javaexception("MyException") METHOD %{
try {
    $action
} catch (MyException) {
    jclass excep = jenv->FindClass("java_throws/MyException");
    if (excep)
        jenv->ThrowNew(excep, "exception message");
    return $null;
}
%}
%enddef

JAVAEXCEPTION(FeatureTest::FeatureTest)
JAVAEXCEPTION(FeatureTest::method)
JAVAEXCEPTION(FeatureTest::staticMethod)

%inline %{
    struct FeatureTest {
        static void staticMethod() {
            throw MyException("no message");
        }
        void method() {
            throw MyException("no message");
        }
    };
%}

// Mixing except feature and typemaps when both generate a class for the throws clause
%typemap(in, throws="ClassNotFoundException") int both { 
    $1 = (int)$input;
}
%javaexception("MyException , NoSuchFieldException") globalFunction %{
try {
    $action
} catch (MyException) {
    jclass excep = jenv->FindClass("java_throws/MyException");
    if (excep)
        jenv->ThrowNew(excep, "exception message");
    return $null;
}
%}

%inline %{
    void globalFunction(int both) {
        throw MyException("no message");
    }
%}

// Test %nojavaexception
%javaexception("MyException") %{
/* global exception handler */
try {
    $action
} catch (MyException) {
    jclass excep = jenv->FindClass("java_throws/MyException");
    if (excep)
        jenv->ThrowNew(excep, "exception message");
    return $null;
}
%}

%nojavaexception *::noExceptionPlease();
%nojavaexception NoExceptTest::NoExceptTest();

// Need to handle the checked exception in NoExceptTest.delete()
%typemap(javafinalize) SWIGTYPE %{
  protected void finalize() {
    try {
      delete();
    } catch (MyException e) {
      throw new RuntimeException(e);
    }
  }
%}

%inline %{
struct NoExceptTest {
  unsigned int noExceptionPlease() { return 123; }
  unsigned int exceptionPlease() { return 456; }
  ~NoExceptTest() {}
};
%}

// Turn global exceptions off (for the implicit destructors/constructors)
%nojavaexception;

