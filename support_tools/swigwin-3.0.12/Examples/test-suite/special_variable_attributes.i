%module special_variable_attributes

// Special variable expansion and special variable macros, aka embedded typemaps - expansion tests
// Tests these are expanded within typemap attributes.
// Also tests that these are expanded when used together, so that the special variables
// can be used as the type passed to the special variable macro.
// C# is used for testing as it is one of the few languages that uses a lot of typemap attributes.
// Attributes in both 'in' and 'out' typemaps are needed, that is,
// typemaps targeting both parameters and return values respectively).

#ifdef SWIGCSHARP
// Check special variable expansion in typemap attributes.
// This changes return by reference into return by value.
// Takes advantage of the fact that 'int' is a valid type in both C and C#.
// This is not a realistic use, just a way to test the variable expansion in the 'out' attribute.
%typemap(ctype, out="$*1_ltype") int& getNumber1 "_not_used_"
%typemap(imtype, out="$*1_ltype") int& getNumber1 "_not_used_"
%typemap(cstype, out="$*1_ltype") int& getNumber1 "_not_used_"
%typemap(out) int& getNumber1 "$result = *$1;"
%typemap(csout, excode=SWIGEXCODE) int & getNumber1 {
    int ret = $imcall;$excode
    return ret;
  }
#endif

%inline %{
int& getNumber1() {
  static int num = 111;
  return num;
}
%}

#ifdef SWIGCSHARP
// Check special variable macro expansion in typemap attributes.
// This changes return by reference into return by value.
%typemap(ctype, out="$typemap(ctype, int)") int& getNumber2 "_not_used_"
%typemap(imtype, out="$typemap(imtype, int)") int& getNumber2 "_not_used_"
%typemap(cstype, out="$typemap(cstype, int)") int& getNumber2 "_not_used_"
%typemap(out) int& getNumber2 "$result = *$1;"
%typemap(csout, excode=SWIGEXCODE) int & getNumber2 {
    int ret = $imcall;$excode
    return ret;
  }
#endif

%inline %{
int& getNumber2() {
  static int num = 222;
  return num;
}
%}


#ifdef SWIGCSHARP
// Check special variable macro expansion and special variable expansion in typemap attributes.
// This changes return by reference into return by value.
%typemap(ctype, out="$typemap(ctype, $*1_ltype)") int& getNumber3 "_not_used_"
%typemap(imtype, out="$typemap(imtype, $*1_ltype)") int& getNumber3 "_not_used_"
%typemap(cstype, out="$typemap(cstype, $*1_ltype)") int& getNumber3 "_not_used_"
%typemap(out) int& getNumber3 "$result = *$1;"
%typemap(csout, excode=SWIGEXCODE) int & getNumber3 {
    int ret = $imcall;$excode
    return ret;
  }
#endif

%inline %{
int& getNumber3() {
  static int num = 333;
  return num;
}
%}

#ifdef SWIGCSHARP
// Check special variable macro expansion in typemap attributes.
%typemap(csin,
         pre="    $typemap(cstype, int) $csinput_scaled = 11;"
        ) int num1
%{$csinput * $csinput_scaled %}
#endif

%inline %{
int bounceNumber1(int num1) {
  return num1;
}
%}

#ifdef SWIGCSHARP
// Check special variable expansion in typemap attributes.
%typemap(csin,
         pre="    $1_type $csinput_scaled = 22;"
        ) int num2
%{$csinput * $csinput_scaled %}
#endif

%inline %{
int bounceNumber2(int num2) {
  return num2;
}
%}

#ifdef SWIGCSHARP
// Check special variable and special variable macro expansion in typemap attributes.
%typemap(csin,
         pre="    $typemap(cstype, $1_type) $csinput_scaled = 33;"
        ) int num3
%{$csinput * $csinput_scaled %}
#endif

%inline %{
int bounceNumber3(int num3) {
  return num3;
}
%}

/////////////////////////////////
//// Multi-argument typemaps ////
/////////////////////////////////

// Test expansion of special variables
#ifdef SWIGCSHARP
%typemap(ctype) (int intvar, char charvar) "double"
%typemap(imtype) (int intvar, char charvar) "double"
%typemap(cstype) (int intvar, char charvar) "double"
%typemap(in) (int intvar, char charvar)
%{
  // split double value a.b into two numbers, a and b*100
  $1 = (int)$input;
  $2 = (char)(($input - $1 + 0.005) * 100);
%}
%typemap(csin,
         pre="    $1_type $csinput_$1_type = 50;\n" // $1_type should expand to int
             "    $2_type $csinput_$2_type = 'A';"  // $2_type should expand to char
        ) (int intvar, char charvar)
%{$csinput + ($csinput_int - 50 + $csinput_char - 'A') + ($csinput_$1_type - 50 + $csinput_$2_type - 'A')%}
#endif

%inline %{
int multi1(int intvar, char charvar) {
  return intvar + charvar;
}
%}

#ifdef SWIGCSHARP
%typemap(csin,
         pre="    $typemap(cstype, int) $csinput_$typemap(cstype, int) = 50;\n" // also should expand to int
             "    $typemap(cstype, char) $csinput_$typemap(cstype, char) = 'A';"  // also should expand to char
        ) (int intvar, char charvar)
%{55 + $csinput + ($csinput_int - 50 + $csinput_char - 'A') + ($csinput_$typemap(cstype, int) - 50 + $csinput_$typemap(cstype, char) - 'A')%}
#endif

%inline %{
int multi2(int intvar, char charvar) {
  return intvar + charvar;
}
%}

#ifdef SWIGCSHARP
%typemap(csin,
         pre="    $typemap(cstype, $1_type) $csinput_$typemap(cstype, $1_type) = 50;\n" // also should expand to int
             "    $typemap(cstype, $2_type) $csinput_$typemap(cstype, $2_type) = 'A';"  // also should expand to char
        ) (int intvar, char charvar)
%{77 + $csinput + ($csinput_int - 50 + $csinput_char - 'A') + ($csinput_$typemap(cstype, $1_type) - 50 + $csinput_$typemap(cstype, $2_type) - 'A')%}
#endif

%inline %{
int multi3(int intvar, char charvar) {
  return intvar + charvar;
}
%}
