%module(directors="1") "director::nestedmodule"

%{
#include <string>

class Foo {
  public:
    virtual ~Foo() {}
    virtual std::string ping() { return "Foo::ping()"; }
    virtual std::string pong() { return "Foo::pong();" + ping(); }

    static Foo* get_self(Foo *slf) {return slf;}
};

%}

%include <std_string.i>

%feature("director") Foo;


class Foo {
  public:
    virtual ~Foo();
    virtual std::string ping();
    virtual std::string pong();

    static Foo* get_self(Foo *slf);
};
