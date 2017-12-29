// This module tests default constructor generation under a
// number of different conditions

%module(ruby_minherit="1") default_constructor

%warnfilter(SWIGWARN_JAVA_MULTIPLE_INHERITANCE,
	    SWIGWARN_CSHARP_MULTIPLE_INHERITANCE,
	    SWIGWARN_D_MULTIPLE_INHERITANCE,
	    SWIGWARN_PHP_MULTIPLE_INHERITANCE) EB; /* C#, D, Java, PHP multiple inheritance */

%warnfilter(SWIGWARN_JAVA_MULTIPLE_INHERITANCE,
	    SWIGWARN_CSHARP_MULTIPLE_INHERITANCE,
	    SWIGWARN_D_MULTIPLE_INHERITANCE,
	    SWIGWARN_PHP_MULTIPLE_INHERITANCE) AD; /* C#, D, Java, PHP multiple inheritance */

%warnfilter(SWIGWARN_JAVA_MULTIPLE_INHERITANCE,
	    SWIGWARN_CSHARP_MULTIPLE_INHERITANCE,
	    SWIGWARN_D_MULTIPLE_INHERITANCE,
	    SWIGWARN_PHP_MULTIPLE_INHERITANCE) GGG; /* C#, D, Java, PHP multiple inheritance */

%warnfilter(SWIGWARN_JAVA_MULTIPLE_INHERITANCE,
	    SWIGWARN_CSHARP_MULTIPLE_INHERITANCE,
	    SWIGWARN_D_MULTIPLE_INHERITANCE,
	    SWIGWARN_PHP_MULTIPLE_INHERITANCE) HHH; /* C#, D, Java, PHP multiple inheritance */

%warnfilter(SWIGWARN_LANG_FRIEND_IGNORE) F; /* friend function */

%delobject F::destroy;
%delobject G::destroy;

%inline %{

/* A class with a public default constructor */
class A {
public:
   A() { };
};

/* This class should get default constructor/destructors */
class AA : public A {
};

/* A class with a public constructor, but not default */

class B {
private:
   B() { }
public:
   B(int x, int y) { }
};

/* This class should get no default constructor, but a destructor */
class BB : public B {
};

/* A class with a protected constructor */
class C {
protected:
    C() { };
public:
};

/* This class does get a default constructor/destructor */
class CC : public C {
};


/* A class with a private constructor */
class D {
private:
   D() { };
public:
   void foo() { };
};

/* This class does not get a default constructor */
class DD: public D {
	
};

/* No default constructor.  A is okay, but D is not */
class AD: public A, public D {

};

/* This class has a default constructor because of optional arguments */
class E {
public:
   E(int x = 0, int y = 0) { }
};

/* This should get a default constructor */
class EE : public E {
};

/* This class should not get a default constructor. B doesn't have one */

class EB : public E, public B {

};

/* A class with a private destructor */

class F {
private:
   ~F() { }
public:
   void foo(int, int) { }
   friend void bar(F *);
   void destroy() { delete this; }
    
};

void bar(F *) { }

#if defined(_MSC_VER)
  #pragma warning(disable: 4624) // destructor could not be generated because a base class destructor is inaccessible or deleted
#endif

// Single inheritance, base has private destructor
class FFF : public F { 
};

// Multiple inheritance, one base has private destructor
class GGG : public A, public F {
};
class HHH : public F, public A {
};

#if defined(_MSC_VER)
  #pragma warning(default: 4624) // destructor could not be generated because a base class destructor is inaccessible or deleted
#endif

/* A class with a protected destructor */
class G {
protected:
   ~G() { }

public:
  static void destroy(G *g) { delete g; }
};

class GG : public G { 
};

template <class T>
class HH_T 
{


public:

  HH_T(int i,int j)
  {
  }
  

protected:
  HH_T();
  
};
 
 
%}
 

%template(HH) HH_T<int>;


%{
  class OSRSpatialReferenceShadow {
  private:
    OSRSpatialReferenceShadow();
  public:
  };
%}

typedef void OSRSpatialReferenceShadow; 

class OSRSpatialReferenceShadow {
private:
public:
  %extend {
    OSRSpatialReferenceShadow( char const * wkt = "" ) {
      return 0;
    }
  } 
};

%inline %{
#ifdef SWIGPYTHON_BUILTIN
bool is_python_builtin() { return true; }
#else
bool is_python_builtin() { return false; }
#endif
%}
  
