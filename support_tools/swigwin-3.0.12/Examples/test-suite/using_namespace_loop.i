%module using_namespace_loop

%inline {
namespace A {
struct Foo;
}

namespace B {
using namespace A;
}

namespace A {
using namespace B;
typedef Foo Bar;
}
}
