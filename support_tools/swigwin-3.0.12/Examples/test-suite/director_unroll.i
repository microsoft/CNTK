%module(directors="1") director_unroll
%{
#include <string>

class Foo {
public:
	virtual ~Foo() {}
	virtual std::string ping() { return "Foo::ping()"; }
	virtual std::string pong() { return "Foo::pong();" + ping(); }
};

class Bar {
private:
	Foo *foo;
public:
	void set(Foo *x) { foo = x; }
	Foo *get() { return foo; }
};

%}

%include "std_string.i"

%feature("director") Foo;

class Foo {
public:
	virtual ~Foo() {}
	virtual std::string ping() { return "Foo::ping()"; }
	virtual std::string pong() { return "Foo::pong();" + ping(); }
};

class Bar {
private:
	Foo *foo;
public:
	void set(Foo *x) { foo = x; }
	Foo *get() { return foo; }
};

