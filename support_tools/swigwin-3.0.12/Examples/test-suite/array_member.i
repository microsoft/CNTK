%module array_member

#if defined(SWIGSCILAB)
%rename(RayPkt) RayPacketData;
#endif

%inline %{

typedef struct Foo {
    char   text[8]; 
    int    data[8];
} Foo;

int global_data[8] = { 0,1,2,3,4,5,6,7 };

void set_value(int *x, int i, int v) {
    x[i] = v;
}

int get_value(int *x, int i) {
    return x[i];
}
%}






#ifdef __cplusplus
%inline 
{

  struct Material
  {
  };

  class RayPacketData {
  public:
    enum {
      Size = 32
    };
    
    const Material * chitMat[Size];
    Material hitMat_val[Size];
    Material *hitMat[Size];

    const Material * chitMat2[Size][Size];
    Material hitMat_val2[Size][Size];
    Material *hitMat2[Size][Size];
  };
}

#endif



%inline %{
#define BUFF_LEN 12

typedef unsigned char BUFF[BUFF_LEN]; 

typedef BUFF MY_BUFF;

typedef struct _m {
  int i;
  MY_BUFF x;
} MyBuff;


typedef char SBUFF[BUFF_LEN];
typedef SBUFF MY_SBUFF;
typedef struct _sm {
  int i;
  MY_SBUFF x;
} MySBuff;

%}
