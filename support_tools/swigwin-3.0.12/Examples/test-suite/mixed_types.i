%module mixed_types

%warnfilter(SWIGWARN_RUBY_WRONG_NAME) hi; /* Ruby, wrong constant name */
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) hello; /* Ruby, wrong constant name */

%warnfilter(SWIGWARN_GO_NAME_CONFLICT);                       /* Ignoring 'hello' due to Go name ('Hello') conflict with 'Hello' */

%inline 
{
  const void* ref_pointer(const void*& a) {
    return a;
  }

  struct A
  {
  };
  
  const A* ref_pointer(A* const& a) {
    return a;
  }

  const A** ref_pointer_1(const A**& a) {
    return a;
  }

  A* pointer_1(A* a) {
    return a;
  }

  const A& ref_const(const A& a) {
    return a;
  }

  enum Hello { hi,hello };

  int sint(int a) {
    return a;
  }

  const int& ref_int(const int& a) {
    return a;
  }

  Hello senum(Hello a) {
    return a;
  }
  
  const Hello& ref_enum(const Hello& a) {
    return a;
  }  

  typedef A *Aptr;
  const Aptr& rptr_const(const Aptr& a) {
    return a;
  }

  const Aptr& rptr_const2(const Aptr& a) {
    return a;
  }

  const void*& rptr_void(const void*& a) {
    return a;
  }

  const A& cref_a(const A& a) {
    return a;
  }

  A& ref_a(A& a) {
    return a;
  }


  template <class T> struct NameT {
  };
  
  
  typedef char name[8];
  typedef char namea[];

  typedef NameT<char> name_t[8];
  
  char* test_a(char hello[8],
	       char hi[],
	       const char chello[8],
	       const char chi[]) {
    return hi;
  }

  int test_b(name n2) {
    return 1;
  }

/* gcc doesn't like this one. Removing until reason resolved.*/
  int test_c(const name& n1) {
    return 1;
  }

  int test_d(name* n1) {
    return 1;
  }

  int test_e(const name_t& n1) {
    return 1;
  }

  int test_f(name_t n1) {
    return 1;
  }

  int test_g(name_t* n1) {
    return 1;
  }

  struct Foo 
  {
    int foo(const Aptr&a);
    int foon(const char (&a)[8]);
  };

  inline int Foo::foo(A* const& a) { return 1; }

}

%{
  inline int Foo::foon(const name& a) { return a[0]; }
%}



%inline %{
#define ARRAY_LEN_X 2
#define ARRAY_LEN_Y 4

typedef enum {One, Two, Three, Four, Five} finger;

typedef struct {
    double         double_field;
} SimpleStruct;

char           array_c [ARRAY_LEN_X][ARRAY_LEN_Y];
signed char    array_sc[ARRAY_LEN_X][ARRAY_LEN_Y];
unsigned char  array_uc[ARRAY_LEN_X][ARRAY_LEN_Y];
short          array_s [ARRAY_LEN_X][ARRAY_LEN_Y];
unsigned short array_us[ARRAY_LEN_X][ARRAY_LEN_Y];
int            array_i [ARRAY_LEN_X][ARRAY_LEN_Y];
unsigned int   array_ui[ARRAY_LEN_X][ARRAY_LEN_Y];
long           array_l [ARRAY_LEN_X][ARRAY_LEN_Y];
unsigned long  array_ul[ARRAY_LEN_X][ARRAY_LEN_Y];
long long      array_ll[ARRAY_LEN_X][ARRAY_LEN_Y];
float          array_f [ARRAY_LEN_X][ARRAY_LEN_Y];
double         array_d [ARRAY_LEN_X][ARRAY_LEN_Y];
SimpleStruct   array_struct[ARRAY_LEN_X][ARRAY_LEN_Y];
SimpleStruct*  array_structpointers[ARRAY_LEN_X][ARRAY_LEN_Y];
int*           array_ipointers [ARRAY_LEN_X][ARRAY_LEN_Y];
finger         array_enum[ARRAY_LEN_X][ARRAY_LEN_Y];
finger*        array_enumpointers[ARRAY_LEN_X][ARRAY_LEN_Y];
const int      array_const_i[ARRAY_LEN_X][ARRAY_LEN_Y] = { {10, 11, 12, 13}, {14, 15, 16, 17} };

void fn_taking_arrays(SimpleStruct array_struct[ARRAY_LEN_X][ARRAY_LEN_Y]) {}

int get_2d_array(int (*array)[ARRAY_LEN_Y], int x, int y){
    return array[x][y];
}
%}

