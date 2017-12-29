%module valuewrapper_opaque

/* 
 *  Opaque types
 */

%feature("valuewrapper") C;
class C;

%{
template<typename T> class TemplateClass {
public:
TemplateClass<T>(T a) {}
};

struct B
{
};

class C
{
public:
  C(int){}
};
%}


/*
 * Hint swig that the Opaque type B don't need the value wrapper.
 * This hint is only necessary in very very special cases.
 */
%feature("novaluewrapper") B;
class B;

/*
 * Force swig to use the value wrapper, even when the class
 * has a default constructor, in case you want to save a
 * instance construction.
 * 
 */
%feature("valuewrapper") D;
class D;


%feature("valuewrapper") A;
class A;

%feature("valuewrapper") TemplateClass<A>;
%feature("valuewrapper") TemplateClass<C>;
template<class T> class TemplateClass;

%feature("valuewrapper") BB;
class BB;


%inline %{

struct A 
{
  A(int){}
};

class D {};

class Klass {};


TemplateClass<Klass> getKlass(Klass k) {
  TemplateClass<Klass> t(k);
  return t;
}


TemplateClass<A> getA(A a) {
  TemplateClass<A> t(a);
  return t;
}


TemplateClass<B> getA(B b) {
  TemplateClass<B> t(b);
  return t;
}


TemplateClass<C> getC(C a) {
  TemplateClass<C> t(a);
  return t;
}


TemplateClass<int> getInt(int a) {
  TemplateClass<int> t(a);
  return t;
}

A sgetA(A a) {
  return a;
}

Klass sgetKlass(Klass a) {
  return a;
}

template <class T> 
struct auto_ptr
{
  auto_ptr(T a){}
};

auto_ptr<A> getPtrA(auto_ptr<A> a) {
  return a;
}

B getB(B a) {
  return a;
}

D getD(D a) {
  return a;
}
 
%}

%template() auto_ptr<A>;


/***** Another strange case, member var + opaque, bug #901706 ******/
%{
class BB {
friend class AA;

protected:
	BB(int aa) { this->a = aa; };
	BB() {};
	
	int a;
};
%}
  
%inline %{

class AA {
public:	
	AA(){}
	
	BB innerObj;
};

%}

%{
class Foobar
{
public:
  Foobar()
  {
  }
  
  char *foo_method()
  {
    return 0;
  }
  
};

class Quux
{
public:
  Quux()
  {
  }
  
  Foobar method()
  {
    return Foobar();
  }
  
};
%}

%feature("novaluewrapper") Foobar;
class Foobar;


class Quux {
public:
  Quux();
  
  Foobar method();

  
};


#if defined(SWIGPYTHON) 

/*
  This case can't be fixed by using the valuewrapper feature and the
  old mechanismbut it works fine with the new mechanism 
*/

%{
 
  // Template primitive type, only visible in C++
  template <class T>
  struct Param
  {
    T val;

    // This case is disabled by now
    // Param(T v): val(v) {}

    Param(T v = T()): val(v) {}
    
    operator T() const { return val; }
  };

%}

/*
  Several languages have 'not 100% safe' typemaps, 
  where the following %applies  don't work. 
*/
%apply int { Param<int> };
%apply const int& { const Param<int>& };

%apply double { Param<double> };
%apply const double& { const Param<double>& };

%inline %{

  template <class T>
  T getv(const Param<T>& p) 
  {
    return p.val;
  }

  template <class T>
  Param<T> getp(const T& v)
  {
    return  Param<T>(v);
  }
  
%}
  
%template(getv_i) getv<int>;
%template(getp_i) getp<int>;

%template(getv_d) getv<double>;
%template(getp_d) getp<double>;

#endif
