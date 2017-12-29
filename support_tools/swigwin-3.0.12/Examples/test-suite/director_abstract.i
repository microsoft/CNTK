%module(directors="1") director_abstract
%{
#include <string>

class Foo {
public:
	virtual ~Foo() {}
	virtual std::string ping() = 0;
	virtual std::string pong() { return "Foo::pong();" + ping(); }
};

%}

%include <std_string.i>

%feature("director") Foo;

class Foo {
public:
  virtual ~Foo() {}
  virtual std::string ping() = 0;
  virtual std::string pong() { return "Foo::pong();" + ping(); }
};



%feature("director");

%inline %{
class Example0
{
protected:
  int xsize, ysize;
  
public:
  
  Example0(int x, int y)
    : xsize(x), ysize(y) { }

  Example0() { }

public:
  virtual ~Example0() {}
  
  int  GetXSize() const { return xsize; }
  
  // pure virtual methods that must be overridden
  virtual int Color(unsigned char r, unsigned char g, unsigned char b)  
  {
    return 0;
  }
  

  static int get_color(Example0 *ptr, unsigned char r, 
		       unsigned char g, unsigned char b) {
    return ptr->Color(r, g, b);
  }
};

class Example1
{
protected:
  int xsize, ysize;
  
protected:
  /* this shouldn't be emitted, unless 'dirprot' is used, since they
   is already a public constructor */
  
  Example1(int x, int y)
    : xsize(x), ysize(y) { }

public:
  Example1() { }

public:
  virtual ~Example1() {}
  
  int  GetXSize() const { return xsize; }
  
  // pure virtual methods that must be overridden
  virtual int Color(unsigned char r, unsigned char g, unsigned char b)  = 0;

  static int get_color(Example1 *ptr, unsigned char r, 
		       unsigned char g, unsigned char b) {
    return ptr->Color(r, g, b);
  }
  

};


class Example2
{
protected:
 int xsize, ysize;

protected:
  /* there is no default constructor, hence, all protected constructors
     should be emitted */

  Example2(int x)
  {
  }

  Example2(int x, int y)
    : xsize(x), ysize(y) { }

public:

  virtual ~Example2() {}

  int  GetXSize() const { return xsize; }

  // pure virtual methods that must be overridden
  virtual int Color(unsigned char r, unsigned char g, unsigned char b) = 0;

  static int get_color(Example2 *ptr, unsigned char r, 
		       unsigned char g, unsigned char b) {
    return ptr->Color(r, g, b);
  }
};

class Example4
{
protected:
 int xsize, ysize;

protected:

  Example4()
  {
  }

  /* this is not emitted, unless dirprot is used */
  Example4(int x, int y)
    : xsize(x), ysize(y) { }

public:

  virtual ~Example4() {}

  int  GetXSize() const { return xsize; }

  // pure virtual methods that must be overridden
  virtual int Color(unsigned char r, unsigned char g, unsigned char b) = 0;

  static int get_color(Example4 *ptr, unsigned char r, 
		       unsigned char g, unsigned char b) {
    return ptr->Color(r, g, b);
  }
};

namespace ns 
{
  template <class T>
  class Example3
  {
  protected:
    /* the default constructor is always emitted, even when protected,
        having another public constructor, and 'dirprot' is not used.
        This is just for Java compatibility */
    Example3()
    {
    }

    /* this is no emitted, unless dirprot mode is used */
    Example3(int x) { }

  public:
    
    Example3(int x, int y) { }

    virtual ~Example3() {}
    
    // pure virtual methods that must be overridden
    virtual int Color(unsigned char r, unsigned char g, unsigned char b) = 0;    

    static int get_color(Example3 *ptr, unsigned char r, 
			 unsigned char g, unsigned char b) {
      return ptr->Color(r, g, b);
    }
  };
}    
%}

%template(Example3_i) ns::Example3<int>;


%inline %{
  struct A{
    virtual ~A() {}
    friend  int g(A* obj);    
  protected:
    A(const A&){}
    virtual int f() = 0;
  };
  
  int g(A* obj) {return 1;}

%}
