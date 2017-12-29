%module(naturalvar) naturalvar

#ifdef __cplusplus
%include std_string.i
%inline 
{
  struct Foo
  {
  };
  

  Foo f;
  std::string s;
  struct Bar
  {
    int i;
    Foo f;
    std::string s;
  };
}
#else
%inline 
{
  typedef struct _foo
  {
  }Foo;
  

  Foo f;
  char *s;
  typedef struct _bar
  {
    int i;
    Foo f;
    char *s;
  }  Bar;
}
#endif

    
