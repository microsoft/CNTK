/* File : example.i */
%module example

%{
extern int gcd(int x, int y);
extern int gcdmain(int argc, char *argv[]);
extern int count(char *bytes, int len, char c);
extern void capitalize (char *str, int len);
extern void circle (double cx, double cy);
extern int squareCubed (int n, int *OUTPUT);
%}

%include exception.i
%include typemaps.i

extern int    gcd(int x, int y);

%typemap(arginit) (int argc, char *argv[]) "$2 = 0;";

%typemap(in) (int argc, char *argv[]) {
  AV *tempav;
  SV **tv;
  I32 len;
  int i;
  if (!SvROK($input)) {
    SWIG_exception(SWIG_ValueError,"$input is not an array.");
  }
  if (SvTYPE(SvRV($input)) != SVt_PVAV) {
    SWIG_exception(SWIG_ValueError,"$input is not an array.");
  }
  tempav = (AV*)SvRV($input);
  len = av_len(tempav);
  $1 = (int) len+1;
  $2 = (char **) malloc(($1+1)*sizeof(char *));
  for (i = 0; i < $1; i++) {
    tv = av_fetch(tempav, i, 0);
    $2[i] = (char *) SvPV(*tv,PL_na);
  }
  $2[i] = 0;
}

%typemap(freearg) (int argc, char *argv[]) {
  free($2);
}

extern int gcdmain(int argc, char *argv[]);

%typemap(in) (char *bytes, int len) {
  STRLEN temp;
  $1 = (char *) SvPV($input, temp);
  $2 = (int) temp;
}

extern int count(char *bytes, int len, char c);


/* This example shows how to wrap a function that mutates a string */

%typemap(in) (char *str, int len) {
  STRLEN templen;
  char *temp;
  temp = (char *) SvPV($input,templen);
  $2 = (int) templen;
  $1 = (char *) malloc($2+1);
  memmove($1,temp,$2);
}

/* Return the mutated string as a new object.  */

%typemap(argout) (char *str, int len) {
  if (argvi >= items) {
    EXTEND(sp,1);
  }
  $result = sv_newmortal();
  sv_setpvn((SV*)ST(argvi++),$1,$2);
  free($1);
}   

extern void capitalize(char *str, int len);

/* A multi-valued constraint.  Force two arguments to lie
   inside the unit circle */

%typemap(check) (double cx, double cy) {
   double a = $1*$1 + $2*$2;
   if (a > 1.0) {
	SWIG_exception(SWIG_ValueError,"$1_name and $2_name must be in unit circle");
   }
}

extern void circle(double cx, double cy);


