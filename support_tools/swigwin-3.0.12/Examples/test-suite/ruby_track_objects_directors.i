%module(directors="1") ruby_track_objects_directors

%{
#include <string>
%}

%include "std_string.i";
%feature("director") Foo;

%trackobjects;

%inline %{

class Foo {
public:
	Foo() {}
	virtual ~Foo() {}
	virtual std::string ping() 
	{
		return "Foo::ping()";
	}

	virtual std::string pong()
	{
		return "Foo::pong();" + ping();
	}
};


class Container {
	Foo* foo_;
public:
	Foo* get_foo() 
	{
		return foo_;
	}

	void set_foo(Foo *foo)
	{
		foo_ = foo;
	}
};

%}
