%module template_default_class_parms

%inline %{
namespace Space {
  struct SomeType {};
  struct AnotherType {};
  template<typename C, typename D = SomeType, typename E = int> class Bar {
  public:
    C CType;
    D DType;
    E EType;
    Bar(C c, D d, E e) {}
    C method(C c, D d, E e) { return c; }
  };
  template<typename T = SomeType> class Foo {
  public:
    T TType;
    Foo(T t) {}
    T method(T t) { return t; }
  };
  template<typename T = int> class ATemplate {};
}
%}

// Use defaults
%template(DefaultBar) Space::Bar<double>;
%template(DefaultFoo) Space::Foo<>;

// Don't use all defaults
%template(BarAnotherTypeBool) Space::Bar<Space::AnotherType, bool>;
%template(FooAnotherType) Space::Foo<Space::AnotherType>;

%template() Space::ATemplate<>;


// Github issue #280 segfault
%inline %{
namespace Teuchos {
  class Describable {};
}
namespace KokkosClassic {
  namespace DefaultNode {
    struct DefaultNodeType {};
  }
}

namespace Tpetra {
  template <class LocalOrdinal = int,
            class GlobalOrdinal = LocalOrdinal,
            class Node = KokkosClassic::DefaultNode::DefaultNodeType>
  class Map : public Teuchos::Describable {
  public:
    typedef LocalOrdinal local_ordinal_type;
    typedef GlobalOrdinal global_ordinal_type;
    typedef Node node_type;
    void test_func(LocalOrdinal, GlobalOrdinal, Node) {}
  };
}
%}

#ifdef SWIGJAVA
// Fixes still required for other languages
%template(MapDefaults) Tpetra::Map<>;
#endif

%inline %{
namespace Details {
  template < class LO = ::Tpetra::Map<>::local_ordinal_type,
            class GO = typename ::Tpetra::Map<LO>::global_ordinal_type,
            class NT = typename ::Tpetra::Map<LO, GO>::node_type >
  class Transfer : public Teuchos::Describable {
  public:
    void transfer_func(LO, GO, NT) {}
  };
}
%}

// Below is not resolving correctly yet
//%template(TransferDefaults) Details::Transfer<>;
