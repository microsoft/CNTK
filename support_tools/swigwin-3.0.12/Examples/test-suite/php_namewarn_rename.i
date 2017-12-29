%module php_namewarn_rename

#ifdef SWIGPHP
%warnfilter(SWIGWARN_PARSE_KEYWORD) Empty();
%warnfilter(SWIGWARN_PARSE_KEYWORD) stdClass;
%warnfilter(SWIGWARN_PARSE_KEYWORD) directory;
%warnfilter(SWIGWARN_PARSE_KEYWORD) Hello::empty();
#endif

%ignore prev::operator++;
%extend prev {
  void next() { ++(*self); }
}

%inline %{

  int Exception() { return 13; }

  void Empty() {}

  class stdClass
  {
  };

  class directory
  {
  };

  struct Hello
  {
    void empty() {}
  };

  struct prev {
    prev & operator++() { return *this; }
    prev operator++(int) { return *this; }
  };

%}
