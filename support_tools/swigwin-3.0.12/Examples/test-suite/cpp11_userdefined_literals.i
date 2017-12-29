/* This testcase checks whether SWIG correctly parses the user-defined literals
   introduced in C++11. */
%module cpp11_userdefined_literals

// Unfortunately full declaration is needed for %rename atm, the parameter list cannot be omitted.
%rename(MyRawLiteral)  operator"" _myRawLiteral(const char * value);
%rename(MySuffixIntegral) operator "" _mySuffixIntegral(unsigned long long);
%rename(MySuffixFloat) operator "" _mySuffixFloat(long double);
%rename(MySuffix1) operator "" _mySuffix1(const char * string_values, size_t num_chars);
%rename(MySuffix2) operator "" _mySuffix2(const wchar_t * string_values, size_t num_chars);
%rename(MySuffix3) operator "" _mySuffix3(const char16_t * string_values, size_t num_chars);
%rename(MySuffix4) operator "" _mySuffix4(const char32_t * string_values, size_t num_chars);

%ignore operator "" _myRawLiteralIgnored(const char * value);

%inline %{
#include <iostream>

struct OutputType {
  int val;
  OutputType(int v) : val(v) {}
};

// Raw literal
OutputType operator "" _myRawLiteral(const char * value) { return OutputType(10); }

// Cooked numeric literals
OutputType operator "" _mySuffixIntegral(unsigned long long) { return OutputType(20); }
OutputType operator "" _mySuffixFloat(long double) { return OutputType(30); }

// Cooked string literals
OutputType operator "" _mySuffix1(const char * string_values, size_t num_chars) { return OutputType(100); }
OutputType operator "" _mySuffix2(const wchar_t * string_values, size_t num_chars) { return OutputType(200); }
OutputType operator "" _mySuffix3(const char16_t * string_values, size_t num_chars) { return OutputType(300); }
OutputType operator "" _mySuffix4(const char32_t * string_values, size_t num_chars) { return OutputType(400); }

OutputType operator"" _myRawLiteralIgnored(const char * value) { return OutputType(15); }
%}

%{
// TODO: SWIG cannot parse these
OutputType some_variable_a = 1234_myRawLiteral;

OutputType some_variable_b = 1234_mySuffixIntegral;
OutputType some_variable_c = 3.1416_mySuffixFloat;

OutputType some_variable_d =   "1234"_mySuffix1;
OutputType some_variable_e = u8"1234"_mySuffix1;
OutputType some_variable_f =  L"1234"_mySuffix2;
OutputType some_variable_g =  u"1234"_mySuffix3;
OutputType some_variable_h =  U"1234"_mySuffix4;
%}

