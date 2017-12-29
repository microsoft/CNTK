/*
Two dimension arrays
*/

%module arrays_global_twodim

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

#ifdef __cplusplus
%inline 
{

  struct Material
  {
  };

  enum {
    Size = 32
  };
  
  const Material * chitMat[Size][Size];
  Material hitMat_val[Size][Size];
  Material *hitMat[Size][Size];
}

#endif

