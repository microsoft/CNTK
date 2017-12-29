%module template_typedef_inherit

// Bug 3378145

%include std_string.i

%inline %{
#include <string>                // for std::string

typedef std::string String;

namespace Type {
  template <class T> class TypedInterfaceObject {
  public:
    virtual ~TypedInterfaceObject() {}
  };

  template <class T> class TypedCollectionInterfaceObject : public TypedInterfaceObject<T> {
  public:
    typedef T                                                   ImplementationType;
    typedef typename ImplementationType::ElementType            ImplementationElementType;

    /** Method add() appends an element to the collection */
    void add(const ImplementationElementType & elt) {}
  };

  template <class T> class PersistentCollection {
  public:
    typedef T ElementType;

    /** Method add() appends an element to the collection */
    inline virtual void add(const T & elt) {}
    virtual ~PersistentCollection() {}
  };
}
%}

%template(StringPersistentCollection) Type::PersistentCollection<String>;

%inline %{

namespace Type {
  class DescriptionImplementation : public PersistentCollection<String> {
  public:
    typedef PersistentCollection<String>::ElementType ElementType;
    DescriptionImplementation() {}
  };
}

%}

%template(DescriptionImplementationTypedInterfaceObject)           Type::TypedInterfaceObject<Type::DescriptionImplementation>;
%template(DescriptionImplementationTypedCollectionInterfaceObject) Type::TypedCollectionInterfaceObject<Type::DescriptionImplementation>;

