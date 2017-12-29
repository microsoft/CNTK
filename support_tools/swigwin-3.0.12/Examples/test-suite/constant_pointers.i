/*
This testcase primarily test constant pointers, eg int* const.  Only a getter is expected to be produced when wrapping constant pointer variables. A number of other const issues are also tested.
*/

%module constant_pointers

#if defined(SWIGSCILAB)
%rename(MbrVar) MemberVariablesTest;
%rename(RetVal) ReturnValuesTest;
#endif

%warnfilter(SWIGWARN_TYPEMAP_SWIGTYPELEAK);                   /* memory leak when setting a ptr/ref variable */
%warnfilter(SWIGWARN_TYPEMAP_SWIGTYPELEAK_MSG);               /* Setting a pointer/reference variable may leak memory. */
 

%inline %{

int        GlobalInt;
const int  ConstInt=2;
int*       GlobalIntPtr=&GlobalInt;
int* const GlobalConstIntPtr=&GlobalInt;
#define ARRAY_SIZE 2

class ParametersTest {
public:
    void param1(int* a) {}
    void param2(const int* a) {}
    void param3(int* const a) {}
    void param4(int const a) {}
    void param5(const int a) {}
    void param6(int& a) {}
    void param7(const int& a) {}
    void param8(int const& a) {}
    void param9(int*& a) {}
    void param10(int* const& a) {}
    void param11(const int* const a) {}

    void param_array1(int* a[ARRAY_SIZE]) {}
    void param_array2(const int* a[ARRAY_SIZE]) {}
    void param_array3(int* const a[ARRAY_SIZE]) {}
    void param_array4(int const a[ARRAY_SIZE]) {}
    void param_array5(const int a[ARRAY_SIZE]) {}
    void param_array6(const int* const a[ARRAY_SIZE]) {}
};

class MemberVariablesTest {
public:
    int* member1;
    ParametersTest* member2;
    int* const member3;
    ParametersTest* const member4;

    int* array_member1[ARRAY_SIZE];
    ParametersTest* array_member2[ARRAY_SIZE];
    MemberVariablesTest() : member3(NULL), member4(NULL) {}
private:
  MemberVariablesTest& operator=(const MemberVariablesTest&);
};
void foofunction(const int *const i) {}

typedef int *typedef1, typedef2, *const typedef3;
int int1, int2=2, *int3, *const int4 = &GlobalInt;

int* const global_const_int_ptr_array[ARRAY_SIZE] = { &int1, &int2 };
ParametersTest* const array_member4[ARRAY_SIZE] = { new ParametersTest(), new ParametersTest() };

class ReturnValuesTest {
public:
    typedef1 td1;
    typedef2 td2;
    int int1, int2, *const int3, *int4, array1[ARRAY_SIZE];
    int ret1() {return 5;}
    const int ret2() {return 5;}
    int ret3() {return 5;}
    const int* ret4() {return &ConstInt;}
    int* const ret5() {return &GlobalInt;}

    void ret6(int*& a) {}
    int*& ret7() {return GlobalIntPtr;}
    void ret8(int*const& a) {}
    int*const& ret9() {return GlobalIntPtr;}
    ReturnValuesTest() : int3(NULL) {}
private:
  ReturnValuesTest& operator=(const ReturnValuesTest&);
};

const int* globalRet1() {return &GlobalInt;}
int* const globalRet2() {return &GlobalInt;}

%}


%{
  struct A
  {
  };
%}


%inline 
{
  typedef const A* Acptr;

  Acptr opaque(Acptr aptr) {
    return aptr;
  }

  struct B
  {
    B() : ca() {}
    const A ca;
    A a;
    A* ap;
    const A* cap;
    Acptr acptr;  
  private:
    B& operator=(const B&);
  };

  const B* bar(const B* b) {
    return b;
  }

  B *const& cbar(B *const& b) {
    return b;
  }
}



%{
static int wxEVT_COMMAND_BUTTON_CLICKEDv;
static int *wxEVT_COMMAND_BUTTON_CLICKEDp;
static int **wxEVT_COMMAND_BUTTON_CLICKEDpp = &wxEVT_COMMAND_BUTTON_CLICKEDp;
#if defined(SWIGR)
#undef lang1 /* conflicts with symbol in R internals */
#endif

char lang1[16] = "Hello";
char *langs[] ={ lang1 };
 
     
%}


%inline {
#define EWXWEXPORT_VAR 

  const int* wxEVENT_COMMAND_BUTTON_CLICKEDr = (int*) &wxEVT_COMMAND_BUTTON_CLICKEDv;
  const int* wxEVENT_COMMAND_BUTTON_CLICKEDp = (int*) *wxEVT_COMMAND_BUTTON_CLICKEDpp;
  char **languages1 = &langs[0];
  char **languages2 = (char **)&langs[0];
}

%inline %{
struct Foo {
  const int *icap;
};
const int* icap;
const Foo *cap;
%}



