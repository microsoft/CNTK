%module li_std_string
%include <std_string.i>

#if defined(SWIGUTL)
%apply std::string& INPUT { std::string &input }
%apply std::string& INOUT { std::string &inout }
#endif


%inline %{

std::string test_value(std::string x) {
   return x;
}

const std::string& test_const_reference(const std::string &x) {
   return x;
}

void test_pointer(std::string *x) {
}

std::string *test_pointer_out() {
   static std::string x = "x";
   return &x;
}

void test_const_pointer(const std::string *x) {
}

const std::string *test_const_pointer_out() {
   static std::string x = "x";
   return &x;
}

void test_reference(std::string &x) {
}

std::string& test_reference_out() {
   static std::string x = "test_reference_out message";
   return x;
}

std::string test_reference_input(std::string &input) {
  return input;
}

void test_reference_inout(std::string &inout) {
  inout += inout;
}

#if defined(_MSC_VER)
  #pragma warning(disable: 4290) // C++ exception specification ignored except to indicate a function is not __declspec(nothrow)
#endif

void test_throw() throw(std::string){
  static std::string x = "test_throw message";
  throw x;
}

void test_const_reference_throw() throw(const std::string &){
  static std::string x = "test_const_reference_throw message";
  throw x;
}

void test_pointer_throw() throw(std::string *) {
  throw new std::string("foo");
}

void test_const_pointer_throw() throw(const std::string *) {
  throw new std::string("foo");
}

#if defined(_MSC_VER)
  #pragma warning(default: 4290) // C++ exception specification ignored except to indicate a function is not __declspec(nothrow)
#endif

%}

/* Old way, now std::string is a %naturalvar by default
%apply const std::string& { std::string *GlobalString2, 
                            std::string *MemberString2, 
                            std::string *Structure::StaticMemberString2 };
*/

#ifdef SWIGSCILAB
%rename(St) MemberString;
%rename(Str) MemberString;
%rename(Str2) MemberString2;
%rename(StaticStr) StaticMemberString;
%rename(StaticStr2) StaticMemberString2;
%rename(ConstStr) ConstMemberString;
%rename(ConstStaticStr) ConstStaticMemberString;
#endif

%inline %{
std::string GlobalString;
std::string GlobalString2 = "global string 2";
const std::string ConstGlobalString = "const global string";

struct Structure {
  std::string MemberString;
  std::string MemberString2;
  static std::string StaticMemberString;
  static std::string StaticMemberString2;

  const std::string ConstMemberString;
  static const std::string ConstStaticMemberString;

  Structure() : MemberString2("member string 2"), ConstMemberString("const member string") {}
};
%}

%{
  std::string Structure::StaticMemberString = "static member string";
  std::string Structure::StaticMemberString2 = "static member string 2";
  const std::string Structure::ConstStaticMemberString = "const static member string";
%}


%inline %{
class Foo {
public:
   unsigned long long  test(unsigned long long l)
   {
       return l + 1;
   }
   std::string test(std::string l)
   {
       return l + "1";
   }

   unsigned long long  testl(unsigned long long l)
   {
       return l + 1;
   }

}; 
%}

%inline %{
  std::string stdstring_empty() {
    return std::string();
  }

  char *c_empty() {
    return (char *)"";
  }

  char *c_null() {
    return 0;
  }

  const char *get_null(const char *a) {
    return a == 0 ? a : "non-null";
  }


%}
