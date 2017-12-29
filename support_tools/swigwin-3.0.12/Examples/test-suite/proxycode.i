%module proxycode

%warnfilter(SWIGWARN_PARSE_NAMED_NESTED_CLASS) Proxy4::Proxy4Nested;

#if defined(SWIGJAVA) || defined(SWIGCSHARP) || defined(SWIGD)

%{
struct Proxy1 {};
%}
struct Proxy1 {
%proxycode %{
  public int proxycode1(int i) {
    return i+1;
  }
%}
};

%proxycode %{
  this should be ignored as it is not in scope of a class
%}

%extend Proxy2 {
%proxycode %{
  public int proxycode2a(int i) {
    return i+2;
  }
%}
}

%extend Proxy2 {
%proxycode %{
  public int proxycode2b(int i) {
    return i+2;
  }
%}
}

%inline %{
struct Proxy2 {};
struct Proxy3 {};
struct Proxy4 {
  struct Proxy4Nested {};
};
%}

%extend Proxy3 {
%proxycode %{
  public int proxycode3(int i) {
    return i+3;
  }
%}
}

%extend Proxy4 {
%proxycode %{
  public int proxycode4(int i) {
    return i+4;
  }
%}
}
%extend Proxy4::Proxy4Nested {
%proxycode %{
  public int proxycode4nested(int i) {
    return i+44;
  }
%}
}

%extend TemplateProxy {
%proxycode %{
  public T proxycode5(T i) {
    return i;
  }
%}
}

%extend TemplateProxy<int> {
%proxycode %{
  public int proxycode5(int i, int j) {
    return i+j+55;
  }
%}
}

%inline %{
template <typename T> struct TemplateProxy {};
%}

%template(Proxy5a) TemplateProxy<short>;
%template(Proxy5b) TemplateProxy<int>;

%inline %{
template <typename T> struct TypemapProxy {
  T useT(T t1, T const& t2) {
    return t1+t2;
  }
};
%}

%extend TypemapProxy {
#if defined(SWIGJAVA)
%proxycode %{
  public $javaclassname proxyUseT(long t1,  long t2) {
    $typemap(jstype, unsigned int) tt1 = t1;
    $typemap(jstype, unsigned int const&) tt2 = t2;
    long ret = useT(tt1, tt2);
    if (ret != t1+t2)
      throw new RuntimeException("wrong sum");
    return this;
  }
%}
#elif defined(SWIGCSHARP)
%proxycode %{
  public $csclassname proxyUseT(uint t1,  uint t2) {
    $typemap(cstype, unsigned int) tt1 = t1;
    $typemap(cstype, unsigned int const&) tt2 = t2;
    uint ret = useT(tt1, tt2);
    if (ret != t1+t2)
      throw new System.Exception("wrong sum");
    return this;
  }
%}
#elif defined(SWIGD)
%proxycode %{
  public $dclassname proxyUseT(uint t1,  uint t2) {
    $typemap(dtype, unsigned int) tt1 = t1;
    $typemap(dtype, unsigned int const&) tt2 = t2;
    uint ret = useT(tt1, tt2);
    if (ret != t1+t2)
      throw new Exception("wrong sum");
    return this;
  }
%}
#else
#error "missing test"
#endif
}

%template(Proxy6) TypemapProxy<unsigned int>;

#endif
