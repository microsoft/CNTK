%module memberin_extend

// Tests memberin typemap is not used for %extend.
// The test extends the struct with a pseudo member variable

%inline %{
#include <string>
struct ExtendMe {
};
%}

%{
#include <map>
std::map<ExtendMe*, char *> ExtendMeStringMap;
void ExtendMe_thing_set(ExtendMe *self, const char *val) {
  char *old_val = ExtendMeStringMap[self];
  delete [] old_val;
  if (val) {
    ExtendMeStringMap[self] = new char[strlen(val)+1];
    strcpy(ExtendMeStringMap[self], val);
  } else {
    ExtendMeStringMap[self] = 0;
  }
}
char * ExtendMe_thing_get(ExtendMe *self) {
  return ExtendMeStringMap[self];
}
%}

%extend ExtendMe {
  char *thing;
}

