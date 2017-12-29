/* File : example.i */
%module dynamic_cast

#if !defined(SWIGJAVA) && !defined(SWIGCSHARP) && !defined(SWIGGO) && !defined(SWIGD)
%apply SWIGTYPE *DYNAMIC { Foo * };
#endif

%inline %{

class Foo {
public:
  virtual ~Foo() { }
  
  virtual Foo *blah() {
    return this;
  }
};
%}

#if defined(SWIGJAVA) || defined(SWIGCSHARP) || defined(SWIGGO) || defined(SWIGD)
%typemap(out) Foo *blah {
    Bar *downcast = dynamic_cast<Bar *>($1);
    *(Bar **)&$result = downcast;
}
#endif

#if defined(SWIGJAVA)
%typemap(javaout) Foo * {
    return new Bar($jnicall, $owner);
  }
#endif

#if defined(SWIGCSHARP)
%typemap(csout, excode=SWIGEXCODE) Foo * {
    Bar ret = new Bar($imcall, $owner);$excode
    return ret;
  }
#endif

#if defined(SWIGD)
%typemap(dout, excode=SWIGEXCODE) Foo * {
  Bar ret = new Bar($imcall, $owner);$excode
  return ret;
}
#endif

#if defined(SWIGGO)
%insert(go_runtime) %{
func FooToBar(f Foo) Bar {
	return SwigcptrBar(f.Swigcptr())
}
%}
#endif

%inline %{

class Bar : public Foo {
public:
   virtual Foo *blah() {
       return (Foo *) this;
   }
   virtual char * test() {
       return (char *) "Bar::test";
   }
};

char *do_test(Bar *b) {
   return b->test();
}
%}

#if !defined(SWIGJAVA) && !defined(SWIGCSHARP) && !defined(SWIGGO) && !defined(SWIGD)
// A general purpose function for dynamic casting of a Foo *
%{
static swig_type_info *
Foo_dynamic(void **ptr) {
   Bar *b;
   b = dynamic_cast<Bar *>((Foo *) *ptr);
   if (b) {
      *ptr = (void *) b;
      return SWIGTYPE_p_Bar;
   }
   return 0;
}
%}

// Register the above casting function
DYNAMIC_CAST(SWIGTYPE_p_Foo, Foo_dynamic);

#endif

