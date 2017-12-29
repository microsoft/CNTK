%module memberin1

%{
class String {
private:
  char *str;
public:
  // Constructor
  String(const char *s = 0) : str(0) {
    if (s != 0) {
      str = new char[strlen(s) + 1];
      strcpy(str, s);
    }
  }

  // Copy constructor
  String(const String& other) {
    delete [] str;
    str = 0;
    if (other.str != 0) {
      str = new char[strlen(other.str) + 1];
      strcpy(str, other.str);
    }
  }

  // Assignment operator
  String& operator=(const String& other) {
    if (&other != this) {
      delete [] str;
      str = 0;
      if (other.str != 0) {
        str = new char[strlen(other.str) + 1];
        strcpy(str, other.str);
      }
    }
    return *this;
  }

  // String contents
  const char *c_str() const { return str; }

  // Destructor
  ~String() { delete [] str; }
};
%}

#ifdef SWIGRUBY
%typemap(in) String {
  Check_Type($input, T_STRING);
  $1 = String(StringValuePtr($input));
}
#endif

%typemap(memberin) String {
  $1 = $input;
}

%inline %{
struct Person {
  String name;
};
%}

