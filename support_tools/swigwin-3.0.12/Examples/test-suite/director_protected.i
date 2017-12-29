%module(directors="1",dirprot="1") director_protected
%{
#include <string>
#include <iostream>
%}

%include "std_string.i"

%feature("director") Foo;
%feature("director") Bar;

%newobject *::create();

#ifdef SWIGPHP
// TODO: Currently we do not track the dynamic type of returned objects
// in PHP, so we need the factory helper.
%include factory.i
%factory(Foo *Bar::create, Bar);
#endif

%rename(a) Bar::hello;
%rename(s) Foo::p;
%rename(q) Foo::r;

%inline {
class Foo {
public:
  virtual ~Foo() {}
  virtual std::string pong() {
    return "Foo::pong();" + ping();
  }

  int p(){ return 1;}
  int r(){ return 1;}
    
  
protected:
  
  typedef int q(); 
  static int s(); 
  
  Foo() {}  

  virtual std::string ping() = 0;

  virtual std::string pang() 
  {
    return "Foo::pang();"; 
  }

  void hellom() {}

  virtual std::string used() {
    return pang() + pong();
  }

  virtual std::string cheer() {
    return pang() + pong();
  }
};

class Bar : public Foo 
{
public:
  Foo* create() 
  {
    return new Bar();
  }

  std::string callping() {
    return ping();
  }

  std::string callcheer() {
    return cheer();
  }

  std::string pong() {
    return "Bar::pong();" + Foo::pong();
  }

  int hello;

  using Foo::used;
  
protected:
  std::string ping() { 
    return "Bar::ping();"; 
  };
  using Foo::cheer;

  enum Hello {hola, chao};

  static int a;
  static const int b;
  
  int hi;
  void him() {}

private:
  int c;

};
 

class PrivateFoo : private Foo 
{
};

}


%director A;
%director B;

%inline %{
  class A {
  public:
    A() {};
    virtual ~A() {};
  protected:
    virtual void draw() {};
  };

  class B : public A {
  public:
    B() {};
    virtual ~B() {};
  protected:
    void draw() {};
    void draw(int arg1) {};
  };

%}


%cleardirector;

%inline %{
  class AA {
  public:
    AA() {};
    virtual ~AA() {};
  protected:
    virtual void draw() {};
    virtual void plot() {};
  };

  class BB : public AA {
  public:
    BB() {};
    virtual ~BB() {};
  protected:
    void draw() {};
    void draw(int arg1) {};

    void plot(int arg1) {};
    void plot() {};
  };
%}

