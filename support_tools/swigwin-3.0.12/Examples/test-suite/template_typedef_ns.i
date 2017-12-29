%module("templatereduce") template_typedef_ns

%inline {
  namespace Alpha {
    typedef int Integer;
  }
  
  namespace Beta {
    template <typename Value>
    struct Alpha {
      Value x;
    };
  }
}


%template(AlphaInt) Beta::Alpha<Alpha::Integer>;

