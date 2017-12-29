%module typemap_variables

// Check typemap name matching rules for variables
// Some of these are using qualified names, which is not right... the test will be adjusted as these get fixed

#if defined(SWIGUTL)
%{
#define TYPEMAP_VARIABLES_FAIL SWIG_fail;
%}
#else
%{
#define TYPEMAP_VARIABLES_FAIL
%}
#endif

// For Javascript V8 we can not use '0' for out typemaps
#if defined(SWIG_JAVASCRIPT_V8)
%header %{
#define OUT_NULL_VALUE SWIGV8_NULL()
%}
#else
%header %{
#define OUT_NULL_VALUE 0
%}
#endif

// Scripting languages use varin/varout for variables (except non-static member variables where in/out are used ???)
%typemap(varin)  int                           "this_will_not_compile_varin "
%typemap(varout) int                           "this_will_not_compile_varout"
%typemap(varin)  int globul                    "/*int globul varin */ TYPEMAP_VARIABLES_FAIL"
%typemap(varout) int globul                    "/*int globul varout*/ $result=OUT_NULL_VALUE;"
%typemap(varin)  int Space::nspace             "/*int nspace varin */ TYPEMAP_VARIABLES_FAIL"
%typemap(varout) int Space::nspace             "/*int nspace varout*/ $result=OUT_NULL_VALUE;"
//%typemap(varin)  int member                    "/*int member varin */"
//%typemap(varout) int member                    "/*int member varout*/ $result=OUT_NULL_VALUE;"
%typemap(varin)  int Space::Struct::smember    "/*int smember varin */ TYPEMAP_VARIABLES_FAIL"
%typemap(varout) int Space::Struct::smember    "/*int smember varout*/ $result=OUT_NULL_VALUE;"

// Statically typed languages use in/out for variables
%typemap(in)  int                           "this_will_not_compile_in "
%typemap(out) int                           "this_will_not_compile_out"
%typemap(in)  int globul                    "/*int globul in */ $1=0;"
%typemap(out) int globul                    "/*int globul out*/ $result=OUT_NULL_VALUE;"
%typemap(in)  int Space::nspace             "/*int nspace in */ $1=0;"
%typemap(out) int Space::nspace             "/*int nspace out*/ $result=OUT_NULL_VALUE;"
%typemap(in)  int member                    "/*int member in */ $1=0;"
#ifdef SWIGTCL
%typemap(out) int member                    "/*int member out*/"
#else
%typemap(out) int member                    "/*int member out*/ $result=OUT_NULL_VALUE;"
#endif
%typemap(in)  int Space::Struct::smember    "/*int smember in */ $1=0;"
%typemap(out) int Space::Struct::smember    "/*int smember out*/ $result=OUT_NULL_VALUE;"

%typemap(javain)  int                           "this_will_not_compile_javain "
%typemap(javaout) int                           "this_will_not_compile_javaout"
%typemap(javain)  int globul                    "/*int globul in */  $javainput"
%typemap(javaout) int globul                    "/*int globul out*/  { return $jnicall; }"
%typemap(javain)  int Space::nspace             "/*int nspace in */  $javainput"
%typemap(javaout) int Space::nspace             "/*int nspace out*/  { return $jnicall; }"
%typemap(javain)  int member                    "/*int member in */  $javainput"
%typemap(javaout) int member                    "/*int member out*/  { return $jnicall; }"
%typemap(javain)  int Space::Struct::smember    "/*int smember in */ $javainput"
%typemap(javaout) int Space::Struct::smember    "/*int smember out*/ { return $jnicall; }"

#if defined(SWIGSCILAB)
%clear int globul;
%clear int Space::nspace;
%clear int Space::Struct::smember;
%ignore Space::Struct::member;
%typemap(varin) int globul "TYPEMAP_VARIABLES_FAIL";
%typemap(varout, noblock=1, fragment=SWIG_From_frag(int)) int globul "if (!SWIG_IsOK(SWIG_Scilab_SetOutput(pvApiCtx, SWIG_From_int($result)))) return SWIG_ERROR;";
%typemap(varin) int Space::nspace "TYPEMAP_VARIABLES_FAIL";
%typemap(varout, noblock=1, fragment=SWIG_From_frag(int)) int Space::nspace "if (!SWIG_IsOK(SWIG_Scilab_SetOutput(pvApiCtx, SWIG_From_int($result)))) return SWIG_ERROR;";
%typemap(varin) int Space::Struct::smember "TYPEMAP_VARIABLES_FAIL";
%typemap(varout, noblock=1, fragment=SWIG_From_frag(int)) int Space::Struct::smember "if (!SWIG_IsOK(SWIG_Scilab_SetOutput(pvApiCtx, SWIG_From_int($result)))) return SWIG_ERROR;";
#endif

%inline %{

int globul;

namespace Space {
  int nspace;
  struct Struct {
    int member;
    static int smember;
//    static short memberfunction() { return 0; } //javaout and jstype typemaps don't use fully qualified name, but other typemaps do
  };
  int Struct::smember = 0;
}
%}

