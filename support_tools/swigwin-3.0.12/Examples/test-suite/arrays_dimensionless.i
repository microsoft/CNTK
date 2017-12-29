
%module arrays_dimensionless

%warnfilter(SWIGWARN_TYPEMAP_VARIN_UNDEF) globalints;    /* Unable to set variable of type int [] */
%warnfilter(SWIGWARN_TYPEMAP_VARIN_UNDEF) ints;          /* Unable to set variable of type int [] */

%inline %{

int globalints[] = {100, 200, 300};
const int constglobalints[] = {400, 500, 600};

struct Bar {
    static int ints[];
};
int Bar::ints[] = {700, 800, 900};

double arr_bool(bool array[], int length)                { double sum=0.0; int i=0; for(; i<length; i++) { sum += array[i]; array[i]=!array[i]; } return sum; }
double arr_char(char array[], int length)                { double sum=0.0; int i=0; for(; i<length; i++) { sum += array[i]; array[i]*=2; } return sum; }
double arr_schar(signed char array[], int length)        { double sum=0.0; int i=0; for(; i<length; i++) { sum += array[i]; array[i]*=2; } return sum; }
double arr_uchar(unsigned char array[], int length)      { double sum=0.0; int i=0; for(; i<length; i++) { sum += array[i]; array[i]*=2; } return sum; }
double arr_short(short array[], int length)              { double sum=0.0; int i=0; for(; i<length; i++) { sum += array[i]; array[i]*=2; } return sum; }
double arr_ushort(unsigned short array[], int length)    { double sum=0.0; int i=0; for(; i<length; i++) { sum += array[i]; array[i]*=2; } return sum; }
double arr_int(int array[], int length)                  { double sum=0.0; int i=0; for(; i<length; i++) { sum += array[i]; array[i]*=2; } return sum; }
double arr_uint(unsigned int array[], int length)        { double sum=0.0; int i=0; for(; i<length; i++) { sum += array[i]; array[i]*=2; } return sum; }
double arr_long(long array[], int length)                { double sum=0.0; int i=0; for(; i<length; i++) { sum += array[i]; array[i]*=2; } return sum; }
double arr_ulong(unsigned long array[], int length)      { double sum=0.0; int i=0; for(; i<length; i++) { sum += array[i]; array[i]*=2; } return sum; }
double arr_ll(long long array[], int length)             { double sum=0.0; int i=0; for(; i<length; i++) { sum += array[i]; array[i]*=2; } return sum; }
double arr_ull(unsigned long long array[], int length)   { double sum=0.0; int i=0; for(; i<length; i++) { sum += array[i]; array[i]*=2; } return sum; }
double arr_float(float array[], int length)              { double sum=0.0; int i=0; for(; i<length; i++) { sum += array[i]; array[i]*=2; } return sum; }
double arr_double(double array[], int length)            { double sum=0.0; int i=0; for(; i<length; i++) { sum += array[i]; array[i]*=2; } return sum; }

%}

%apply SWIGTYPE[] {
 bool *, 
 char *, 
 signed char *, 
 unsigned char *, 
 short *, 
 unsigned short *, 
 int *, 
 unsigned int *, 
 long *, 
 unsigned long *, 
 long *, 
 unsigned long long *, 
 float *, 
 double *
}

%inline %{

double ptr_bool(bool *array, int length)                { double sum=0.0; int i=0; for(; i<length; i++) sum += array[i]; return sum; }
double ptr_char(char *array, int length)                { double sum=0.0; int i=0; for(; i<length; i++) sum += array[i]; return sum; }
double ptr_schar(signed char *array, int length)        { double sum=0.0; int i=0; for(; i<length; i++) sum += array[i]; return sum; }
double ptr_uchar(unsigned char *array, int length)      { double sum=0.0; int i=0; for(; i<length; i++) sum += array[i]; return sum; }
double ptr_short(short *array, int length)              { double sum=0.0; int i=0; for(; i<length; i++) sum += array[i]; return sum; }
double ptr_ushort(unsigned short *array, int length)    { double sum=0.0; int i=0; for(; i<length; i++) sum += array[i]; return sum; }
double ptr_int(int *array, int length)                  { double sum=0.0; int i=0; for(; i<length; i++) sum += array[i]; return sum; }
double ptr_uint(unsigned int *array, int length)        { double sum=0.0; int i=0; for(; i<length; i++) sum += array[i]; return sum; }
double ptr_long(long *array, int length)                { double sum=0.0; int i=0; for(; i<length; i++) sum += array[i]; return sum; }
double ptr_ulong(unsigned long *array, int length)      { double sum=0.0; int i=0; for(; i<length; i++) sum += array[i]; return sum; }
double ptr_ll(long long *array, int length)             { double sum=0.0; int i=0; for(; i<length; i++) sum += array[i]; return sum; }
double ptr_ull(unsigned long long *array, int length)   { double sum=0.0; int i=0; for(; i<length; i++) sum += array[i]; return sum; }
double ptr_float(float *array, int length)              { double sum=0.0; int i=0; for(; i<length; i++) sum += array[i]; return sum; }
double ptr_double(double *array, int length)            { double sum=0.0; int i=0; for(; i<length; i++) sum += array[i]; return sum; }

%}

