%module abstract_typedef2

 /*
   After the fix for abstract_typedef, this simpler
   example got broken.
 */

%inline %{

  enum FieldDim {
    UnaryField,
    BinaryField
  };
  
  template <FieldDim Dim>
    struct Facet;
  

  template <FieldDim Dim>
    struct Base 
    {
      virtual ~Base() {}
      
      typedef unsigned int size_type;
      typedef Facet<Dim>* facet_ptr;

      // This works 
      // virtual Facet<Dim>* set(size_type) = 0;
      
      // This doesn't
      virtual facet_ptr set(size_type) = 0;
    };
  

  template <FieldDim Dim>
    struct Facet
    {
    };
  

  template <FieldDim Dim>
    struct A : Base<Dim>
    {
      typedef Base<Dim> base;
      typedef typename base::size_type size_type;

      A(int a = 0)
      {
      }
      
      Facet<Dim>* set(size_type) 
      {
	return 0;
      }      
    };
%}


%template(Base_UF) Base<UnaryField>;
%template(A_UF) A<UnaryField>;
