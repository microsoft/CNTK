/* Tests the Java specific directives */

%module java_typemaps_proxy


%typemap(javaimports) SWIGTYPE "import java.math.*;";
%typemap(javacode) NS::Farewell %{
  public void saybye(BigDecimal num_times) {
    // BigDecimal requires the java.math library
  }
%}
%typemap(javaclassmodifiers) NS::Farewell "public final class";

%typemap(javaimports) NS::Greeting %{
import java.util.*; // for EventListener
import java.lang.*; // for Exception
%};

%typemap(javabase) NS::Greeting "Exception";
%typemap(javainterfaces) NS::Greeting "EventListener";
%typemap(javacode) NS::Greeting %{
  public static final long serialVersionUID = 0x52151000; // Suppress ecj warning
  // Pure Java code generated using %typemap(javacode) 
  public void sayhello() {
    hello();
  }

  public static void cheerio(EventListener e) {
  }
%}

// Create a new getCPtr() function which takes Java null and is public
%typemap(javabody) NS::Greeting %{
  private transient long swigCPtr;
  protected transient boolean swigCMemOwn;

  protected $javaclassname(long cPtr, boolean cMemoryOwn) {
    swigCMemOwn = cMemoryOwn;
    swigCPtr = cPtr;
  }

  public static long getCPtr($javaclassname obj) {
    return (obj == null) ? 0 : obj.swigCPtr;
  }
%}

// Make the pointer constructor public
%typemap(javabody) NS::Farewell %{
  private transient long swigCPtr;
  protected transient boolean swigCMemOwn;

  public $javaclassname(long cPtr, boolean cMemoryOwn) {
    swigCMemOwn = cMemoryOwn;
    swigCPtr = cPtr;
  }

  protected static long getCPtr($javaclassname obj) {
    return (obj == null) ? 0 : obj.swigCPtr;
  }
%}

// get rid of the finalize method for NS::Farewell
%typemap(javafinalize) NS::Farewell "";

// Test typemaps are being found for templated classes
%typemap(javacode) NS::Adieu<int**> %{
  public static void adieu() {
  }
%}

// Check the %javamethodmodifiers feature
%javamethodmodifiers NS::Farewell::methodmodifiertest() "private";

%inline %{
namespace NS {
    class Greeting {
    public:
        void hello() {}
        static void ciao(Greeting* g) {}
    };
    class Farewell {
    public:
        void methodmodifiertest() {}
    };
    template<class T> class Adieu {};
}
%}

%template(AdieuIntPtrPtr) NS::Adieu<int**>;

// Check the premature garbage collection prevention parameter can be turned off
%typemap(jtype, nopgcpp="1") Without * "long";
%pragma(java) jniclassclassmodifiers="public class"

%inline %{
struct Without {
  Without(Without *p) : var(0) {}
  static void static_method(Without *p) {}
  void member_method(Without *p) {}
  Without *var;
};
Without *global_without = 0;
void global_method_without(Without *p) {}
struct With {
  With(With *p) {}
  static void static_method(With *p) {}
  void member_method(With *p) {}
};
void global_method_with(With *p) {}
%}

%typemap(jtype, nopgcpp="1") const ConstWithout * "long";
%inline %{
class ConstWithout {
public:
  ConstWithout(const ConstWithout *p) : const_var(0), var_const(0) {}
  static void static_method(const ConstWithout *p) {}
  void member_method(const ConstWithout *p) {}
  void const_member_method(const ConstWithout *p) const {}
  const ConstWithout * const_var;
  const ConstWithout * const var_const;
private:
  ConstWithout& operator=(const ConstWithout &);
};
const ConstWithout * global_constwithout = 0;
void global_method_constwithout(const ConstWithout *p) {}
%}


