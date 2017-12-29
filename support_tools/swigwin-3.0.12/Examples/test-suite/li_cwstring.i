%module li_cwstring

%include "cwstring.i"

#ifndef SWIG_CWSTRING_UNIMPL

%cwstring_input_binary(wchar_t *str_in, int n);
%cwstring_bounded_output(wchar_t *out1, 512);
%cwstring_chunk_output(wchar_t *out2, 64);
%cwstring_bounded_mutable(wchar_t *out3, 512);
%cwstring_mutable(wchar_t *out4, 32);
%cwstring_output_maxsize(wchar_t *out5, int max);
%cwstring_output_withsize(wchar_t *out6, int *size);

#ifdef __cplusplus
%cwstring_output_allocate(wchar_t **out7, delete [] *$1);
%cwstring_output_allocate_size(wchar_t **out8, int *size, delete [] *$1);
#else
%cwstring_output_allocate(wchar_t **out7, free(*$1));
%cwstring_output_allocate_size(wchar_t **out8, int *size, free(*$1));
#endif

%inline %{

int count(wchar_t *str_in, int n, wchar_t c) {
   int r = 0;
   while (n > 0) {
     if (*str_in == c) {
	r++;
     }
     str_in++;
     --n;
   }
   return r;
}

void test1(wchar_t *out1) {
   wcscpy(out1,L"Hello World");
}

void test2(wchar_t *out2) {
   int i;
   for (i = 0; i < 64; i++) {
       *out2 = (wchar_t) i + 32;
       out2++;
   }
}

void test3(wchar_t *out3) {
   wcscat(out3,L"-suffix");
}

void test4(wchar_t *out4) {
   wcscat(out4,L"-suffix");
}

void test5(wchar_t *out5, int max) {
   int i;
   for (i = 0; i < max; i++) {
       out5[i] = 'x';
   }
   out5[max]='\0';
}

void test6(wchar_t *out6, int *size) {
   int i;
   for (i = 0; i < (*size/2); i++) {
       out6[i] = 'x';
   }
   *size = (*size/2);
}

void test7(wchar_t **out7) {
#ifdef __cplusplus
   *out7 = new wchar_t[64];
#else
   *out7 = malloc(64*sizeof(wchar_t));
#endif
   (*out7)[0] = 0;
   wcscat(*out7,L"Hello world!");
}

void test8(wchar_t **out8, int *size) {
   int i;
#ifdef __cplusplus
   *out8 = new wchar_t[64];
#else
   *out8 = malloc(64*sizeof(wchar_t));
#endif
   for (i = 0; i < 64; i++) {
      (*out8)[i] = (wchar_t) i + 32;
   }
   *size = 64;
}

%}

#endif
