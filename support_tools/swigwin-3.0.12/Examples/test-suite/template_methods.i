// Test %ignore and %rename for templated methods

%module template_methods

%warnfilter(SWIGWARN_LANG_TEMPLATE_METHOD_IGNORE) convolve1<float>();
%warnfilter(SWIGWARN_LANG_TEMPLATE_METHOD_IGNORE) convolve3<float>();

%include <std_string.i>

///////////////////
%ignore convolve1<float>(float a);

%inline %{
template <typename ImageT> int convolve1() { return 0; }
template <typename ImageT> void convolve1(ImageT a) { ImageT t = a; (void)t; }
%}

%template() convolve1<float>;
%template(convolve1Bool) convolve1<bool>;


///////////////////
%ignore convolve2<float>(float a);

%inline %{
template <typename ImageT> int convolve2() { return 0; }
template <typename ImageT> void convolve2(ImageT a) { ImageT t = a; (void)t; }
%}

%template(convolve2Float) convolve2<float>;

///////////////////
%rename(convolve3FloatRenamed) convolve3<float>(float a);

%inline %{
template <typename ImageT> int convolve3() { return 0; }
template <typename ImageT> void convolve3(ImageT a) { ImageT t = a; (void)t; }
%}

%template() convolve3<float>;

///////////////////
%rename(convolve4FloatRenamed) convolve4<float>(float a);

%inline %{
template <typename ImageT> int convolve4() { return 0; }
template <typename ImageT> void convolve4(ImageT a) { ImageT t = a; (void)t; }
%}

%template(convolve4Float) convolve4<float>;


///////////////////
%rename(convolve5FloatRenamed) convolve5<float>;
%ignore convolve5<bool>;

%inline %{
template <typename ImageT> int convolve5() { return 0; }
template <typename ImageT> void convolve5(ImageT a) { ImageT t = a; (void)t; }
%}

%template() convolve5<float>;
%template() convolve5<bool>;


////////////////////////////////////////////////////////////////////////////
%rename(KlassTMethodBoolRenamed) Klass::tmethod(bool);
%rename(KlassStaticTMethodBoolRenamed) Klass::statictmethod(bool);

%inline %{
struct Klass {
  template<typename X> X tmethod(X x) { return x; }
  template<typename X> void tmethod() {}
  template<typename X> static X statictmethod(X x) { return x; }
  template<typename X> static void statictmethod() {}
};
%}
%template(KlassTMethodBool) Klass::tmethod<bool>;
%template(KlassStaticTMethodBool) Klass::statictmethod<bool>;

////////////////////////////////////////////////////////////////////////////

%inline %{
  class ComponentProperties{
  public:
    ComponentProperties() {}
    ~ComponentProperties() {}

    template <typename T1> void adda(std::string key, T1 val) {}
    template <typename T1, typename T2> void adda(std::string key1, T1 val1, std::string key2, T2 val2) {}
    template <typename T1, typename T2, typename T3> void adda(std::string key1, T1 val1, std::string key2, T2 val2, std::string key3, T3 val3) {}
  };
%}

%extend ComponentProperties {
  %template(adda) adda<std::string, double>;
  %template(adda) adda<std::string, std::string, std::string>; // ERROR OCCURS HERE
  %template(adda) adda<int, int, int>;
}

