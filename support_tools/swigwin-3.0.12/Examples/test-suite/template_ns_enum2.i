%module template_ns_enum2
 
%inline %{ 
  
  namespace hi 
  { 
    enum Hello { Hi, Hola }; 
    
    template <Hello> 
    struct traits 
    { 
    }; 
 
    template <> 
    struct traits<Hi> 
    { 
      typedef int value_type; 
    }; 
 
    template <> 
    struct traits<Hola> 
    { 
      typedef double value_type; 
    }; 
    
    struct B 
    { 
      const traits<Hola>::value_type& 
          evaluate(const traits<Hola>::value_type& e) const; 
    
    }; 
    const traits<Hola>::value_type&
    B::evaluate(const traits<Hola>::value_type& e) const {
      return e;
    }
  } 
 
%} 
