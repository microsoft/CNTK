%module(ruby_minherit="1") evil_diamond_prop

%warnfilter(SWIGWARN_RUBY_WRONG_NAME) foo;		// Ruby, wrong class name
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) bar;		// Ruby, wrong class name
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) baz;		// Ruby, wrong class name
%warnfilter(SWIGWARN_RUBY_WRONG_NAME,
	    SWIGWARN_JAVA_MULTIPLE_INHERITANCE,
	    SWIGWARN_CSHARP_MULTIPLE_INHERITANCE,
	    SWIGWARN_D_MULTIPLE_INHERITANCE,
	    SWIGWARN_PHP_MULTIPLE_INHERITANCE) spam;	// Ruby, wrong class name - C# & Java, PHP multiple inheritance

%inline %{

class foo { 
  public:
    int _foo;
    foo() : _foo(1) {}
};

class bar : public foo {
  public:
    int _bar;
    bar() : _bar(2) {}
};

class baz : public foo {
  public:
    int _baz;
    baz() : _baz(3) {}
};

class spam : public bar, public baz {
  public:
    int _spam;
    spam() : _spam(4) {}
};

foo *test(foo *f) { return f; }
%}

