%module template_keyword_in_type

%warnfilter(SWIGWARN_PARSE_NAMED_NESTED_CLASS) YYY;

%inline %{
  template<typename T> struct XXX;

  template<typename T> struct XXX {
    template<typename TT> struct YYY {
      typedef TT type;
    };
    int xxx(int h) { return h; }
  };

#if defined(SWIG) || defined(__clang__)
  // gcc doesn't parse this (tested with gcc-4.8)
  template<typename X> void testXXX1(::template XXX<int>::template YYY<int>::type xx) {}
#else
  template<typename X> void testXXX1(::         XXX<int>::template YYY<int>::type xx) {}
#endif
  template<typename X> void testXXX2(XXX<int>::YYY<int>::type xx) {}
  template<typename X> void testXXX3(::XXX<int>::template YYY<int>::type) {}
%}

%inline %{
typedef double Val;
template<typename T> struct ListBucket {
};
namespace Alloc {
  template<typename T> struct rebind {
    typedef int other;
  };
}

template<typename X> void other1(typename Alloc::template rebind<ListBucket<Val> >::other) {}
#if !defined(__SUNPRO_CC)
template<typename X> void other2(typename Alloc::template rebind< ::template ListBucket<Val> >::other) {}
#else
template<typename X> void other2(typename Alloc::template rebind< ::         ListBucket<Val> >::other) {}
#endif
template<typename X> void other3(Alloc::template rebind<int>) {}
%}
