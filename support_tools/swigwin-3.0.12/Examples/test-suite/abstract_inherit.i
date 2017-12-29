%module abstract_inherit

%warnfilter(SWIGWARN_TYPE_ABSTRACT) Spam;
%warnfilter(SWIGWARN_TYPE_ABSTRACT) Bar;

%inline %{

class Foo {
public:
  virtual ~Foo()
  {
  }
  
  virtual int blah() = 0;
};

class Bar : public Foo { };

class Spam: public Foo {
public:
   Spam() { }
};


template <class Type>
class NRFilter {
public:
  virtual ~NRFilter()
  {
  }
  
protected:
  virtual void do_filter() = 0;
};

template <class Type>
class NRRCFilter : public NRFilter<Type>
{
};
 

template <class Type>
class NRRCFilterpro : protected NRFilter<Type>
{
};


template <class Type>
class NRRCFilterpri : private NRFilter<Type>
{
};

%}

%template(NRFilter_i) NRFilter<int>;

%template(NRRCFilter_i) NRRCFilter<int>;

%template(NRRCFilterpro_i) NRRCFilterpro<int>;

%template(NRRCFilterpri_i) NRRCFilterpri<int>;

