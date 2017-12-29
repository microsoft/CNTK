/*
A test case for testing char based strings. Useful for runtime testing,
there were some runtime crashes and leaks in the C# module in some of the scenarios
below.
*/

%module char_strings

%warnfilter(SWIGWARN_TYPEMAP_VARIN_UNDEF) global_char_array1;  // Unable to set variable of type char[]
%warnfilter(SWIGWARN_TYPEMAP_CHARLEAK_MSG) global_const_char;  // Setting a const char * variable may leak memory.

%{
#include <stdio.h>

#define OTHERLAND_MSG "Little message from the safe world."
#define CPLUSPLUS_MSG "A message from the deep dark world of C++, where anything is possible."
static char *global_str = NULL;
const int UINT_DIGITS = 10; // max unsigned int is 4294967295

bool check(const char *const str, unsigned int number) {
  static char expected[256];
  sprintf(expected, "%s%d", OTHERLAND_MSG, number);
  bool matches = (strcmp(str, expected) == 0);
  if (!matches) printf("Failed: [%s][%s]\n", str, expected);
  return matches;
}

%}

%immutable global_const_char;

%inline %{
// get functions
char *GetCharHeapString() {
  global_str = new char[sizeof(CPLUSPLUS_MSG)+1];
  strcpy(global_str, CPLUSPLUS_MSG);
  return global_str;
}

const char *GetConstCharProgramCodeString() {
  return CPLUSPLUS_MSG;
}

void DeleteCharHeapString() {
  delete[] global_str;
  global_str = NULL;
}

char *GetCharStaticString() {
  static char str[sizeof(CPLUSPLUS_MSG)+1];
  strcpy(str, CPLUSPLUS_MSG);
  return str;
}

char *GetCharStaticStringFixed() {
  static char str[] = CPLUSPLUS_MSG;
  return str;
}

const char *GetConstCharStaticStringFixed() {
  static const char str[] = CPLUSPLUS_MSG;
  return str;
}

// set functions
bool SetCharHeapString(char *str, unsigned int number) {
  delete[] global_str;
  global_str = new char[strlen(str)+UINT_DIGITS+1];
  strcpy(global_str, str);
  return check(global_str, number);
}

bool SetCharStaticString(char *str, unsigned int number) {
  static char static_str[] = CPLUSPLUS_MSG;
  strcpy(static_str, str);
  return check(static_str, number);
}

bool SetCharArrayStaticString(char str[], unsigned int number) {
  static char static_str[] = CPLUSPLUS_MSG;
  strcpy(static_str, str);
  return check(static_str, number);
}

bool SetConstCharHeapString(const char *str, unsigned int number) {
  delete[] global_str;
  global_str = new char[strlen(str)+UINT_DIGITS+1];
  strcpy(global_str, str);
  return check(global_str, number);
}

bool SetConstCharStaticString(const char *str, unsigned int number) {
  static char static_str[] = CPLUSPLUS_MSG;
  strcpy(static_str, str);
  return check(static_str, number);
}

bool SetConstCharArrayStaticString(const char str[], unsigned int number) {
  static char static_str[] = CPLUSPLUS_MSG;
  strcpy(static_str, str);
  return check(static_str, number);
}

bool SetCharConstStaticString(char *const str, unsigned int number) {
  static char static_str[] = CPLUSPLUS_MSG;
  strcpy(static_str, str);
  return check(static_str, number);
}

bool SetConstCharConstStaticString(const char *const str, unsigned int number) {
  static char static_str[] = CPLUSPLUS_MSG;
  strcpy(static_str, str);
  return check(static_str, number);
}

// get set function
char *CharPingPong(char *str) {
  return str;
}
char *CharArrayPingPong(char abcstr[]) {
  return abcstr;
}
char *CharArrayDimsPingPong(char abcstr[16]) {
  return abcstr;
}

// variables
char *global_char = NULL;
char global_char_array1[] = CPLUSPLUS_MSG;
char global_char_array2[sizeof(CPLUSPLUS_MSG)+1] = CPLUSPLUS_MSG;

const char *global_const_char = CPLUSPLUS_MSG;
const char global_const_char_array1[] = CPLUSPLUS_MSG;
const char global_const_char_array2[sizeof(CPLUSPLUS_MSG)+1] = CPLUSPLUS_MSG;

%}


%typemap(newfree) char *GetNewCharString() { /* hello */ delete[] $1; }
%newobject GetNewCharString();

%inline {
  char *GetNewCharString() {
    char *nstr = new char[sizeof(CPLUSPLUS_MSG)+1];
    strcpy(nstr, CPLUSPLUS_MSG);
    return nstr;
  }
}

%inline {
  struct Formatpos;
  struct OBFormat;
  
  static int GetNextFormat(Formatpos& itr, const  char*& str,OBFormat*& pFormat) {
    return 0;
  }
  


}


%inline %{

// char *& tests
char *&GetCharPointerRef() {
  static char str[] = CPLUSPLUS_MSG;
  static char *ptr = str;
  return ptr;
}

bool SetCharPointerRef(char *&str, unsigned int number) {
  static char static_str[] = CPLUSPLUS_MSG;
  strcpy(static_str, str);
  return check(static_str, number);
}

const char *&GetConstCharPointerRef() {
  static const char str[] = CPLUSPLUS_MSG;
  static const char *ptr = str;
  return ptr;
}

bool SetConstCharPointerRef(const char *&str, unsigned int number) {
  static char static_str[] = CPLUSPLUS_MSG;
  strcpy(static_str, str);
  return check(static_str, number);
}

%}

