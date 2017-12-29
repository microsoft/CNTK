%module csharp_typemaps

// Test the C# types customisation by modifying the default char * typemaps to return a single char

%typemap(ctype, out="char /*ctype out override*/") char * "char *"
%typemap(imtype, out="char /*imtype out override*/") char * "string"
%typemap(cstype, out="char /*cstype out override*/") char * "string"

%typemap(out) char * %{
  // return the 0th element rather than the whole string
  $result = SWIG_csharp_string_callback($1)[0];
%}

%typemap(csout, excode=SWIGEXCODE) char * {
    char ret = $imcall;$excode
    return ret;
  }

%typemap(csvarout, excode=SWIGEXCODE2) char * %{
    get {
      char ret = $imcall;$excode
      return ret;
    } %}

%inline %{
namespace Space {
    class Things {
    public:
        char* start(char *val) { return val; }
        static char* stop(char *val) { return val; }
    };
    char* partyon(char *val) { return val; }
}
%}


// Test variables when ref is used in the cstype typemap - the variable name should come from the out attribute if specified
%typemap(cstype) MKVector, const MKVector& "MKVector"
%typemap(cstype, out="MKVector") MKVector &, MKVector * "ref MKVector"

%inline %{
struct MKVector {
};
struct MKRenderGameVector {
  MKVector memberValue;
  static MKVector staticValue;
};
MKVector MKRenderGameVector::staticValue;
MKVector globalValue;
%}


// Number and Obj are for the eager garbage collector runtime test
%inline %{
struct Number {
  Number(double value) : Value(value) {}
  double Value;
};

class Obj {
public:
  Number triple(Number n) {
    n.Value *= 3;
    return n;
  }
  Number times6(const Number& num) {
    Number n(num);
    n.Value *= 6;
    return n;
  }
  Number times9(const Number* num) {
    Number n(*num);
    n.Value *= 9;
    return n;
  }
};
Number quadruple(Number n) {
    n.Value *= 4;
    return n;
}
Number times8(const Number& num) {
    Number n(num);
    n.Value *= 8;
    return n;
}
Number times12(const Number* num) {
    Number n(*num);
    n.Value *= 12;
    return n;
}
%}

// Test $csinput expansion
%typemap(csvarin, excode=SWIGEXCODE2) int %{
    set {
      if ($csinput < 0)
        throw new global::System.ApplicationException("number too small!");
      $imcall;$excode
    } %}

%inline %{
int myInt = 0;
%}


// Illegal special variable crash
%typemap(cstype) WasCrashing "$csclassname /*cstype $*csclassname*/" // $*csclassname was causing crash
%inline %{
struct WasCrashing {};
void hoop(WasCrashing was) {}
%}


// Enum underlying type
%typemap(csbase) BigNumbers "uint"
%inline %{
enum BigNumbers { big=0x80000000, bigger };
%}

// Member variable qualification
%typemap(cstype) bool "badtype1"
%typemap(cstype) bool mvar "badtype2"
%typemap(cstype) bool svar "badtype4"
%typemap(cstype) bool gvar "badtype5"
%typemap(cstype) bool MVar::mvar "bool"
%typemap(cstype) bool MVar::svar "bool"
%typemap(cstype) bool Glob::gvar "bool"
%inline %{
struct MVar {
  bool mvar;
  static bool svar;
};
namespace Glob {
  bool gvar;
}
bool MVar::svar = false;
%}

