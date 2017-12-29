%module template_template_parameters


%inline %{
  namespace pfc {
    template<typename t_item, template <typename> class t_alloc> class array_t {};
    template<typename t_item> class alloc_fast {
      public:
        typedef t_item alloc_type;
    };
  }

  template<typename t_item, typename t2> class list_impl_t {};

  template<typename t_item, template<typename> class t_alloc = pfc::alloc_fast >
    class list_tt : public list_impl_t<t_item,pfc::array_t<t_item,t_alloc> > {
  public:
    t_item item;
//    typename t_alloc<t_item>::alloc_type allotype; // SWIG can't handle this yet
    void xx() {
      typename t_alloc<t_item>::alloc_type atype; // this type is the same as t_item type
      atype = true;
    }
  };

void TestInstantiations() {
  pfc::array_t<int, pfc::alloc_fast> myArrayInt;
  list_impl_t<int, pfc::array_t<int, pfc::alloc_fast> > myListImplInt;
  (void) myArrayInt;
  (void) myListImplInt;
}
%}

%template(ListImplFastBool) list_impl_t<bool, pfc::array_t<bool, pfc::alloc_fast> >;
%template(ListFastBool) list_tt<bool, pfc::alloc_fast>;

%template(ListImplFastDouble) list_impl_t<double, pfc::array_t<double, pfc::alloc_fast> >;
%template(ListDefaultDouble) list_tt<double>;

