// Test the goin and goout typemaps for directors.

%module(directors="1") go_director_inout

%{
#include <string>
%}

%inline
%{

struct MyStruct {
  std::string str;
};

struct RetStruct {
  std::string str;
};

%}

%go_import("encoding/json")

%insert(go_header)
%{

type GoRetStruct struct {
	Str string
}

%}

%typemap(gotype) RetStruct "GoRetStruct"

%typemap(imtype) RetStruct "string"

%typemap(goin) RetStruct
%{
	$result = $input.Str
%}

%typemap(in) RetStruct
%{
  $result.str.assign($input.p, $input.n);
%}

%typemap(out,fragment="AllocateString") RetStruct
%{
  $result = Swig_AllocateString($1.str.data(), $1.str.length());
%}

%typemap(goout,fragment="CopyString") RetStruct
%{
	$result = GoRetStruct{Str: swigCopyString($input)}
%}

%typemap(godirectorout) RetStruct
%{
	$result = $input.Str
%}

%typemap(directorout) RetStruct
%{
  $result.str.assign($input.p, $input.n);
%}

%typemap(godirectorin) RetStruct
%{
%}

%typemap(gotype) MyStruct "map[string]interface{}"

%typemap(imtype) MyStruct "string"

%typemap(goin) MyStruct
%{
	if b, err := json.Marshal($input); err != nil {
		panic(err)
	} else {
		$result = string(b)
	}
%}

%typemap(directorin,fragment="AllocateString") MyStruct
%{
  $input = Swig_AllocateString($1.str.data(), $1.str.length());
%}

%typemap(godirectorin,fragment="CopyString") MyStruct
%{
	if err := json.Unmarshal([]byte(swigCopyString($input)), &$result); err != nil {
		panic(err)
	}
%}

%typemap(out,fragment="AllocateString") MyStruct
%{
  $result = Swig_AllocateString($1.str.data(), $1.str.length());
%}

%typemap(goout,fragment="CopyString") MyStruct
%{
	$result = swigCopyString($input)
%}

%typemap(in) MyStruct
%{
  $1.str.assign($input.p, $input.n);
%}

%feature("director") MyClass;

%inline
%{

class MyClass {
 public:
  virtual ~MyClass() {}
  virtual RetStruct Adjust(MyStruct s) {
    RetStruct r;
    r.str = s.str;
    return r;
  }
};

%}
