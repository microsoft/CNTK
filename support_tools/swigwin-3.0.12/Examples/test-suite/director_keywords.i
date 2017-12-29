// Checks if collisions of argument names with target language keywords are
// resolved properly when directors are used
%module(directors="1") director_keywords

%warnfilter(SWIGWARN_PARSE_KEYWORD);

%feature("director") Foo;

%inline %{
struct Foo {
  virtual ~Foo() {}
  virtual void check_abstract(int abstract) {} // for Java, C#, D...
  virtual void check_self(int self) {} // self for Python
  virtual void check_from(int from) {} // for Python
};
%}
