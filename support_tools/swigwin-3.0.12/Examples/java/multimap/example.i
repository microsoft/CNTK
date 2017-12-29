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

extern int    gcd(int x, int y);

%typemap(jni) (int argc, char *argv[]) "jobjectArray"
%typemap(jtype) (int argc, char *argv[]) "String[]"
%typemap(jstype) (int argc, char *argv[]) "String[]"

%typemap(javain) (int argc, char *argv[]) "$javainput"

%typemap(in) (int argc, char *argv[]) (jstring *jsarray) {
int i;

  $1 = (*jenv)->GetArrayLength(jenv, $input);
  if ($1 == 0) {
    SWIG_JavaThrowException(jenv, SWIG_JavaIndexOutOfBoundsException, "Array must contain at least 1 element");
    return $null;
  }
  $2 = (char **) malloc(($1+1)*sizeof(char *));
  jsarray = (jstring *) malloc($1*sizeof(jstring));
  for (i = 0; i < $1; i++) {
    jsarray[i] = (jstring) (*jenv)->GetObjectArrayElement(jenv, $input, i);
    $2[i] = (char *) (*jenv)->GetStringUTFChars(jenv, jsarray[i], 0);
  }
  $2[i] = 0;
}

%typemap(argout) (int argc, char *argv[]) "" /* override char *[] default */

%typemap(freearg) (int argc, char *argv[]) {
int i;
  for (i = 0; i < $1; i++) {
    (*jenv)->ReleaseStringUTFChars(jenv, jsarray$argnum[i], $2[i]);
  }
  free($2);
}

extern int gcdmain(int argc, char *argv[]);

%typemap(jni) (char *bytes, int len) "jstring"
%typemap(jtype) (char *bytes, int len) "String"
%typemap(jstype) (char *bytes, int len) "String"

%typemap(javain) (char *bytes, int len) "$javainput"

%typemap(in) (char *bytes, int len) {
  $1 = ($1_type)(*jenv)->GetStringUTFChars(jenv, $input, 0);
  $2 = (*jenv)->GetStringUTFLength(jenv, $input);
}

%typemap(freearg) (char *bytes, int len) %{
  (*jenv)->ReleaseStringUTFChars(jenv, $input, $1);
%}

extern int count(char *bytes, int len, char c);

/* This example shows how to wrap a function that mutates a c string. A one
 * element Java string array is used so that the string can be returned modified.*/

%typemap(jni) (char *str, int len) "jobjectArray"
%typemap(jtype) (char *str, int len) "String[]"
%typemap(jstype) (char *str, int len) "String[]"

%typemap(javain) (char *str, int len) "$javainput"

%typemap(in) (char *str, int len) (jstring js) {
  int index = 0;
  js = (jstring) (*jenv)->GetObjectArrayElement(jenv, $input, index);
  $1 = (char *) (*jenv)->GetStringUTFChars(jenv, js, 0);
  $2 = (*jenv)->GetStringUTFLength(jenv, js);
}

/* Return the mutated string as a modified element in the array. */
%typemap(argout) (char *str, int len) {
  jstring newstring = (*jenv)->NewStringUTF(jenv, $1);
  (*jenv)->SetObjectArrayElement(jenv, $input, 0, newstring);
}

/* Release memory */
%typemap(freearg) (char *str, int len) {
  (*jenv)->ReleaseStringUTFChars(jenv, js$argnum, $1);
}

extern void capitalize(char *str, int len);

/* A multi-valued constraint.  Force two arguments to lie
   inside the unit circle */

%typemap(check) (double cx, double cy) {
   double a = $1*$1 + $2*$2;
   if (a > 1.0) {
    SWIG_JavaThrowException(jenv, SWIG_JavaIllegalArgumentException, "$1_name and $2_name must be in unit circle");
    return;
   }
}

extern void circle(double cx, double cy);


