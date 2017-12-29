%module(naturalvar="1") implicittest

%implicitconv;

%inline 
{
  struct B { };  
}

%inline 
{
  struct A
  {
    int ii;
    A(int i) { ii = 1; }
    A(double d) { ii = 2; }
    A(const B& b) { ii = 3; }
    explicit A(char *s) { ii = 4; }

    int get() const { return ii; }
  };

  int get(const A& a) { return a.ii; }

  template <class T>
  struct A_T
  {
    int ii;
    A_T(int i) { ii = 1; }
    A_T(double d) { ii = 2; }
    A_T(const B& b) { ii = 3; }
    explicit A_T(char *s) { ii = 4; }

    int get() const { return ii; }
    static int sget(const A_T& a) { return a.ii; }
  };
}

%inline 
{
  struct Foo 
  {
    int ii;
    Foo(){ ii = 0;}
    Foo(int){ ii = 1;}
    Foo(double){ ii = 2;}
    explicit Foo(char *s){ii = 3;}
    Foo(const Foo& f){ ii = f.ii;}
  };

  struct Bar 
  {
    int ii;
    Foo f;
    Bar() {ii = -1;}
    Bar(const Foo& ff){ ii = ff.ii;}
  };

  int get_b(const Bar&b) { return b.ii; }
  
  Foo foo;
}

%template(A_int) A_T<int>;


/****************** None handling *********************/

%inline
{
  struct BB {};
  struct AA
  {
    int ii;
    AA(int i) { ii = 1; }
    AA(double d) { ii = 2; }
    AA(const B* b) { ii = 3; }
    explicit AA(char *s) { ii = 4; }
    AA(const BB& b) { ii = 5; }

    int get() const { return ii; }
  };

  int get_AA_val(AA a) { return a.ii; }
  int get_AA_ref(const AA& a) { return a.ii; }
}


/****************** Overloading priority *********************/

%inline %{
class BBB {
  public:
    BBB(const B &) {}
};

class CCC {
  public:
    CCC(const BBB &) : checkvalue(0) {}
    int xx(int i) { return 11; }
    int xx(const A& i) { return 22; }
    int yy(int i, int j) { return 111; }
    int yy(const A& i, const A& j) { return 222; }
    int checkvalue;
};
%}

// CCC(const BBB &) was being called instead of this constructor (independent of being added via %extend)
%extend CCC {
  CCC(const B& b) {
    CCC* ccc = new CCC(b);
    ccc->checkvalue = 10;
    return ccc;
  }
};

