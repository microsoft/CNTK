// This test is designed for testing wrappers in the target language in a multi-threaded environment.
// The most common cause for this test failing is incorrect compiler settings for a multi-threaded environment.

%module threads

%include "std_string.i"

%newobject Kerfuffle::CharString;

%inline %{
  #include <string>
  struct Kerfuffle {
    std::string StdString(std::string str) {
      return str;
    }
    char * CharString(const char *str) {
      char * retstr = new char[256];
      strcpy(retstr, str);
      return retstr;
    }
  };
%}

