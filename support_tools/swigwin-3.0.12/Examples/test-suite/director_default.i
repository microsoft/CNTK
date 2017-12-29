%module(directors="1") director_default

%warnfilter(SWIGWARN_TYPEMAP_THREAD_UNSAFE,SWIGWARN_TYPEMAP_DIRECTOROUT_PTR) DefaultsBase;
%warnfilter(SWIGWARN_TYPEMAP_THREAD_UNSAFE,SWIGWARN_TYPEMAP_DIRECTOROUT_PTR) DefaultsDerived;

%{
#include <string>

class Foo {
public:
        Foo(int i = -1) {}
	virtual ~Foo() {}
	virtual std::string Msg(std::string msg = "default") { return "Foo-" + msg; }

	std::string GetMsg() { return Msg(); }
	std::string GetMsg(std::string msg) { return Msg(msg); }
};

%}

%include <std_string.i>

%feature("director") Foo;

class Foo {
public:
        Foo(int i = -1) {}
	virtual ~Foo() {}
	virtual std::string Msg(std::string msg = "default") { return msg; }

	std::string GetMsg() { return Msg(); }
	std::string GetMsg(std::string msg) { return Msg(msg); }
};


%inline %{
class Bar {
public:
        Bar() {}
        Bar(int i) {}
	virtual ~Bar() {}
	virtual std::string Msg(std::string msg = "default") { return "Bar-" + msg; }

	std::string GetMsg() { return Msg(); }
	std::string GetMsg(std::string msg) { return Msg(msg); }
};

%}

%feature("director") DefaultsBase;
%feature("director") DefaultsDerived;

%inline %{
typedef int* IntegerPtr;
typedef double Double;

struct DefaultsBase {
	virtual IntegerPtr defaultargs(double d, int * a = 0) = 0;
        virtual ~DefaultsBase() {}
};

struct DefaultsDerived : DefaultsBase {
	int * defaultargs(Double d, IntegerPtr a = 0) { return 0; }
};
%}

