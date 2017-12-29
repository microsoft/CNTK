/*
This test case tests that various types of arrays are working.
*/

%warnfilter(SWIGWARN_TYPEMAP_CHARLEAK,SWIGWARN_TYPEMAP_VARIN_UNDEF);

%module arrays_global

%inline %{
#define ARRAY_LEN 2

typedef enum {One, Two, Three, Four, Five} finger;

typedef struct {
	double         double_field;
} SimpleStruct;

char           array_c [ARRAY_LEN];
signed char    array_sc[ARRAY_LEN];
unsigned char  array_uc[ARRAY_LEN];
short          array_s [ARRAY_LEN];
unsigned short array_us[ARRAY_LEN];
int            array_i [ARRAY_LEN];
unsigned int   array_ui[ARRAY_LEN];
long           array_l [ARRAY_LEN];
unsigned long  array_ul[ARRAY_LEN];
long long      array_ll[ARRAY_LEN];
float          array_f [ARRAY_LEN];
double         array_d [ARRAY_LEN];
SimpleStruct   array_struct[ARRAY_LEN];
SimpleStruct*  array_structpointers[ARRAY_LEN];
int*           array_ipointers [ARRAY_LEN];
finger         array_enum[ARRAY_LEN];
finger*        array_enumpointers[ARRAY_LEN];
const int      array_const_i[ARRAY_LEN] = {10, 20};

%}

%inline %{
  
const char BeginString_FIX44a[8] = "FIX.a.a"; 
char BeginString_FIX44b[8] = "FIX.b.b"; 

const char BeginString_FIX44c[] = "FIX.c.c"; 
char BeginString_FIX44d[] = "FIX.d.d"; 

const char* BeginString_FIX44e = "FIX.e.e"; 
const char* const BeginString_FIX44f = "FIX.f.f"; 

typedef char name[8];
typedef char namea[];

char* test_a(char hello[8],
	     char hi[],
	     const char chello[8],
	     const char chi[]) {
  return hi;
}

char* test_b(name a, const namea b)  {
  return a;
}

int test_a(int a)  {
  return a;
}

int test_b(int a)  {
  return a;
}
 
%}



#ifdef __cplusplus
%inline 
{

  struct Material
  {
  };

  enum {
    Size = 32
  };
  
  const Material * chitMat[Size];
  Material hitMat_val[Size];
  Material *hitMat[Size];
}

#endif
