%module typemap_manyargs

%typemap(in,numinputs=0) (int* a1, int* a2, int* a3, int* a4, int* a5, int* a6, int *a7, int *a8, int *a9, int *a10) (int temp1,int temp2,int temp3,int temp4,int temp5,int temp6,int temp7,int temp8, int temp9, int temp10)
{
  $1 = &temp1;      // the code generate for this is arg2 = &temp1;
  $2 = &temp2;      // the code generate for this is arg3 = &temp2;
  $3 = &temp3;      // and so on...
  $4 = &temp4;
  $5 = &temp5;
  $6 = &temp6;
  $7 = &temp7;
  $8 = &temp8;
  $9 = &temp9;
  $10 = &temp10;   // the code generated for this was arg20 = &temp10; and arg20 does not exist.
  int $10_ptr = 0; // Was arg20_ptr
  (void)$10_ptr;
}

%inline %{
void my_c_function(char * filename,int* a1, int* a2, int* a3, int* a4, int* a5, int* a6, int *a7, int *a8, int *a9, int *a10) {}
%}
