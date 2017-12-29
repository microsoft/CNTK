/* File : operator_overload.i */
/*
This is a test of all the possible operator overloads

see bottom for a set of possible tests
*/
%module operator_overload

#if defined(SWIGPYTHON)
%warnfilter(SWIGWARN_IGNORE_OPERATOR_EQ,
	    SWIGWARN_IGNORE_OPERATOR_INDEX,
	    SWIGWARN_IGNORE_OPERATOR_PLUSPLUS,
	    SWIGWARN_IGNORE_OPERATOR_MINUSMINUS,
	    SWIGWARN_IGNORE_OPERATOR_LAND,
	    SWIGWARN_IGNORE_OPERATOR_LOR);
#endif

#if !defined(SWIGLUA) && !defined(SWIGR)
%rename(Equal) operator =;
%rename(PlusEqual) operator +=;
%rename(MinusEqual) operator -=;
%rename(MultiplyEqual) operator *=;
%rename(DivideEqual) operator /=;
%rename(PercentEqual) operator %=;
%rename(Plus) operator +;
%rename(Minus) operator -;
%rename(Multiply) operator *;
%rename(Divide) operator /;
%rename(Percent) operator %;
%rename(Not) operator !;
%rename(IndexIntoConst) operator[](unsigned idx) const;
%rename(IndexInto) operator[](unsigned idx);
%rename(Functor) operator ();
%rename(EqualEqual) operator ==;
%rename(NotEqual) operator !=;
%rename(LessThan) operator <;
%rename(LessThanEqual) operator <=;
%rename(GreaterThan) operator >;
%rename(GreaterThanEqual) operator >=;
%rename(And) operator &&;
%rename(Or) operator ||;
%rename(PlusPlusPrefix) operator++();
%rename(PlusPlusPostfix) operator++(int);
%rename(MinusMinusPrefix) operator--();
%rename(MinusMinusPostfix) operator--(int);
#endif

%rename(IndexInto) *::operator[](unsigned idx); // some languages have a %rename *::operator[] already in place, which seems to takes precedence over the above %rename operator[].

#ifdef SWIGCSHARP
%csmethodmodifiers operator++() "protected";
%csmethodmodifiers operator++(int) "private";
%csmethodmodifiers operator--() "private";
%csmethodmodifiers operator--(int) "protected";
%typemap(cscode) Op %{
  public static Op operator++(Op op) {
    // Unlike C++, operator++ must not modify the parameter and both prefix and postfix operations call this method
    Op newOp = new Op(op.i);
    newOp.PlusPlusPostfix(0);
    return newOp;
  }
  public static Op operator--(Op op) {
    // Unlike C++, operator-- must not modify the parameter and both prefix and postfix operations call this method
    Op newOp = new Op(op.i);
    newOp.MinusMinusPrefix();
    return newOp;
  }
%}
#endif

#ifdef SWIGPHP
%rename(AndOperator) operator &&;
%rename(OrOperator) operator ||;
#endif

#if defined(SWIGPYTHON)
%feature("python:slot", "tp_str", functype="reprfunc") Op::__str__;
#endif

#ifdef SWIGD
// Due to the way operator overloading is implemented in D1 and D2, the prefix
// increment/decrement operators (D1) resp. the postfix ones (D2) are ignored. 
%warnfilter(SWIGWARN_IGNORE_OPERATOR_PLUSPLUS, SWIGWARN_IGNORE_OPERATOR_MINUSMINUS);
#endif

%rename(IntCast) operator int();
%rename(DoubleCast) operator double();

%inline %{
#include <stdio.h>

#if defined(_MSC_VER)
  #include <iso646.h> /* for named logical operator, eg 'operator or' */
#endif

class Op {
public:
  int i;
  Op(int a=0) : i(a)
  {}
  Op(const Op& o) : i(o.i)
  {}
  virtual ~Op()
  {}

  friend Op operator &&(const Op& a,const Op& b){return Op(a.i&&b.i);}
  friend Op operator or(const Op& a,const Op& b){return Op(a.i||b.i);}

  Op &operator=(const Op& o) {
    i=o.i;
    return *this;
  }
  // +=,-=... are member fns
  Op &operator+=(const Op& o){ i+=o.i; return *this; }
  Op &operator-=(const Op& o){ i-=o.i; return *this; }
  Op &operator*=(const Op& o){ i*=o.i; return *this; }
  Op &operator/=(const Op& o){ i/=o.i; return *this; }
  Op &operator%=(const Op& o){ i%=o.i; return *this; }
  // the +,-,*,... are friends
  // (just to make life harder)
  friend Op operator+(const Op& a,const Op& b){return Op(a.i+b.i);}
  friend Op operator-(const Op& a,const Op& b);
  friend Op operator*(const Op& a,const Op& b){return Op(a.i*b.i);}
  friend Op operator/(const Op& a,const Op& b){return Op(a.i/b.i);}
  friend Op operator%(const Op& a,const Op& b){return Op(a.i%b.i);}

  // unary operators
  Op operator-() const {return Op(-i);}
  bool operator !() const {return !(i);}

  // overloading the [] operator
  // need 2 versions: get & set
  // note: C++ can be a little mixed up upon which version it calls
  // most of the time it calls the second version
  int operator[](unsigned idx)const
  {	  if (idx==0) return i; return 0;}
  int& operator[](unsigned idx)
  {	  if (idx==0) return i; static int j;j=0; return j;}

  // overloading the () operator
  // this can have many parameters so we will test this
  int operator()(int a=0){return i+a;}
  int operator()(int a,int b){return i+a+b;}

  // increment/decrement operators
  Op& operator++() {++i; return *this;} // prefix ++
  Op operator++(int) {Op o = *this; ++(*this); return o;} // postfix ++
  Op& operator--() {--i; return *this;} // prefix --
  Op operator--(int) {Op o = *this; --(*this); return o;} // postfix --

  // TODO: <<,<<=

  // cast operators
  operator double() { return i; }
  virtual operator int() { return i; }

  // This method just checks that the operators are implemented correctly
  static void sanity_check();
};

// just to complicate matters
// we have a couple of non class operators
inline bool operator==(const Op& a,const Op& b){return a.i==b.i;}
inline bool operator!=(const Op& a,const Op& b){return a.i!=b.i;}
inline bool operator< (const Op& a,const Op& b){return a.i<b.i;}
inline bool operator<=(const Op& a,const Op& b){return a.i<=b.i;}
inline bool operator> (const Op& a,const Op& b){return a.i>b.i;}
inline bool operator>=(const Op& a,const Op& b){return a.i>=b.i;}

%}

%{
  // This one is not declared inline as VC++7.1 gets mixed up with the unary operator-
  Op operator-(const Op& a,const Op& b){return Op(a.i-b.i);}
%}

// in order to wrapper this correctly
// we need to extend the class
// to make the friends & non members part of the class
%extend Op{
        Op operator &&(const Op& b){return Op($self->i&&b.i);}
        Op operator or(const Op& b){return Op($self->i||b.i);}

	Op operator+(const Op& b){return Op($self->i+b.i);}
	Op operator-(const Op& b){return Op($self->i-b.i);}
	Op operator*(const Op& b){return Op($self->i*b.i);}
	Op operator/(const Op& b){return Op($self->i/b.i);}
	Op operator%(const Op& b){return Op($self->i%b.i);}

	bool operator==(const Op& b){return $self->i==b.i;}
	bool operator!=(const Op& b){return $self->i!=b.i;}
	bool operator< (const Op& b){return $self->i<b.i;}
	bool operator<=(const Op& b){return $self->i<=b.i;}
	bool operator> (const Op& b){return $self->i>b.i;}
	bool operator>=(const Op& b){return $self->i>=b.i;}

	// subtraction with reversed arguments
	Op __rsub__(const int b){return Op(b - $self->i);}

	// we also add the __str__() fn to the class
	// this allows it to be converted to a string (so it can be printed)
	const char* __str__()
	{
		static char buffer[255];
		sprintf(buffer,"Op(%d)",$self->i);
		return buffer;
	}
	// to get the [] operator working correctly we need to extend with two function
	// __getitem__ & __setitem__
	int __getitem__(unsigned i)
	{	return (*$self)[i];	}
	void __setitem__(unsigned i,int v)
	{	(*$self)[i]=v;	}
}

/*
Suggested list of operator overloads (mainly from python)

Operators overloaded with their C++ equivalent
__add__,__sub__,__mul__,__div__,__mod__	+,-,*,/,%
__iadd__,__isub__,__imul__,__idiv__,__imod__	+=,-=,*=,/=,%=

__eq__,__ne__,__lt__,__le__,__gt__,__ge__ ==,!=,<,<=,>,>=
__not__,__neg__	unary !, unary -
__and__,__or__,__xor__	logical and,logical or,logical xor
__rshift__,__lshift__ >>,<<

__getitem__,__setitem__ for operator[]

Operators overloaded without C++ equivalents
__pow__ for power operator
__str__ converts object to a string (should return a const char*)
__concat__ for concatenation (if language supports)

*/

%inline %{
class OpDerived : public Op {
public:
  OpDerived(int a=0) : Op(a)
  {}

  // overloaded
  virtual operator int() { return i*2; }
};
%}


%{
#include <stdexcept>
#define ASSERT(X) { if (!(X)) { throw std::runtime_error(#X); } }

void Op::sanity_check()
{
	// test routine:
	Op a;
	Op b=5;
	Op c=b;	// copy construct
	Op d=2;
        Op dd=d; // assignment operator

	// test equality
	ASSERT(a!=b);
	ASSERT(b==c);
	ASSERT(a!=d);
        ASSERT(d==dd);

	// test <
	ASSERT(a<b);
	ASSERT(a<=b);
	ASSERT(b<=c);
	ASSERT(b>=c);
	ASSERT(b>d);
	ASSERT(b>=d);

	// test +=
	Op e=3;
	e+=d;
	ASSERT(e==b);
	e-=c;
	ASSERT(e==a);
	e=Op(1);
	e*=b;
	ASSERT(e==c);
	e/=d;
	ASSERT(e==d);
	e%=c;
	ASSERT(e==d);

	// test +
	Op f(1),g(1);
	ASSERT(f+g==Op(2));
	ASSERT(f-g==Op(0));
	ASSERT(f*g==Op(1));
	ASSERT(f/g==Op(1));
	ASSERT(f%g==Op(0));

	// test unary operators
	ASSERT(!a==true);
	ASSERT(!b==false);
	ASSERT(-a==a);
	ASSERT(-b==Op(-5));

	// test []
	Op h=3;
	ASSERT(h[0]==3);
	ASSERT(h[1]==0);
	h[0]=2;	// set
	ASSERT(h[0]==2);
	h[1]=2;	// ignored
	ASSERT(h[0]==2);
	ASSERT(h[1]==0);

	// test ()
	Op i=3;
	ASSERT(i()==3);
	ASSERT(i(1)==4);
	ASSERT(i(1,2)==6);

	// plus add some code to check the __str__ fn
	//ASSERT(str(Op(1))=="Op(1)");
	//ASSERT(str(Op(-3))=="Op(-3)");

        // test ++ and --
        Op j(100);
        int original = j.i;
        {
          Op newOp = j++;
          int newInt = original++;
          ASSERT(j.i == original);
          ASSERT(newOp.i == newInt);
        }
        {
          Op newOp = j--;
          int newInt = original--;
          ASSERT(j.i == original);
          ASSERT(newOp.i == newInt);
        }
        {
          Op newOp = ++j;
          int newInt = ++original;
          ASSERT(j.i == original);
          ASSERT(newOp.i == newInt);
        }
        {
          Op newOp = --j;
          int newInt = --original;
          ASSERT(j.i == original);
          ASSERT(newOp.i == newInt);
        }

        // cast operators
        Op k=3;
        int check_k = k;
        ASSERT (check_k == 3);

        Op l=4;
        double check_l = l;
        ASSERT (check_l == 4);
}

%}

