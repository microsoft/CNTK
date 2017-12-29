%module template_using_directive_and_declaration_forward
// Test using directives combined with using declarations and forward declarations (templates)

%inline %{
namespace Outer1 {
  namespace Space1 {
    template<typename T> class Thing1;
  }
}
using namespace Outer1::Space1;
using Outer1::Space1::Thing1;
#ifdef __clang__
namespace Outer1 {
  namespace Space1 {
    template<typename T> class Thing1 {};
  }
}
#else
template<typename T> class Thing1 {};
#endif
void useit1(Thing1<int> t) {}
void useit1a(Outer1::Space1::Thing1<int> t) {}
void useit1b(::Outer1::Space1::Thing1<int> t) {}
namespace Outer1 {
  void useit1c(Space1::Thing1<int> t) {}
}


namespace Outer2 {
  namespace Space2 {
    template<typename T> class Thing2;
  }
}
using namespace Outer2;
using Space2::Thing2;
#ifdef __clang__
namespace Outer2 {
  namespace Space2 {
    template<typename T> class Thing2 {};
  }
}
#else
template<typename T> class Thing2 {};
#endif
void useit2(Thing2<int> t) {}
void useit2a(Outer2::Space2::Thing2<int> t) {}
void useit2b(::Outer2::Space2::Thing2<int> t) {}
void useit2c(Space2::Thing2<int> t) {}
namespace Outer2 {
  void useit2d(Space2::Thing2<int> t) {}
}


namespace Outer3 {
  namespace Space3 {
    namespace Middle3 {
      template<typename T> class Thing3;
    }
  }
}
using namespace Outer3;
using namespace Space3;
using Middle3::Thing3;
#ifdef __clang__
namespace Outer3 {
  namespace Space3 {
    namespace Middle3 {
      template<typename T> class Thing3 {};
    }
  }
}
#else
template<typename T> class Thing3 {};
#endif
void useit3(Thing3<int> t) {}
void useit3a(Outer3::Space3::Middle3::Thing3<int> t) {}
void useit3b(::Outer3::Space3::Middle3::Thing3<int> t) {}
void useit3c(Middle3::Thing3<int> t) {}
namespace Outer3 {
  namespace Space3 {
    void useit3d(Middle3::Thing3<int> t) {}
  }
}


namespace Outer4 {
  namespace Space4 {
    namespace Middle4 {
      template<typename T> class Thing4;
    }
  }
}
using namespace Outer4::Space4;
using Middle4::Thing4;
#ifdef __clang__
namespace Outer4 {
  namespace Space4 {
    namespace Middle4 {
      template<typename T> class Thing4 {};
    }
  }
}
#else
template<typename T> class Thing4 {};
#endif
void useit4(Thing4<int> t) {}
void useit4a(Outer4::Space4::Middle4::Thing4<int> t) {}
void useit4b(::Outer4::Space4::Middle4::Thing4<int> t) {}
void useit4c(Middle4::Thing4<int> t) {}
namespace Outer4 {
  namespace Space4 {
    void useit4d(Middle4::Thing4<int> t) {}
  }
}


namespace Outer5 {
  namespace Space5 {
    namespace Middle5 {
      namespace More5 {
        template<typename T> class Thing5;
      }
    }
  }
}
using namespace ::Outer5::Space5;
using namespace Middle5;
using More5::Thing5;
#ifdef __clang__
namespace Outer5 {
  namespace Space5 {
    namespace Middle5 {
      namespace More5 {
	template<typename T> class Thing5 {};
      }
    }
  }
}
#else
template<typename T> class Thing5 {};
#endif
void useit5(Thing5<int> t) {}
void useit5a(Outer5::Space5::Middle5::More5::Thing5<int> t) {}
void useit5b(::Outer5::Space5::Middle5::More5::Thing5<int> t) {}
void useit5c(Middle5::More5::Thing5<int> t) {}
namespace Outer5 {
  namespace Space5 {
    void useit5d(Middle5::More5::Thing5<int> t) {}
  }
}

namespace Outer7 {
  namespace Space7 {
    namespace Middle7 {
      template<typename T> class Thing7;
    }
  }
}
using namespace Outer7::Space7;
#ifdef __clang__
namespace Outer7 {
  namespace Space7 {
    namespace Middle7 {
      template<typename T> class Thing7 {};
    }
  }
}
#else
template<typename T> class Middle7::Thing7 {};
#endif
using Middle7::Thing7;
void useit7(Thing7<int> t) {}
void useit7a(Outer7::Space7::Middle7::Thing7<int> t) {}
void useit7b(::Outer7::Space7::Middle7::Thing7<int> t) {}
void useit7c(Middle7::Thing7<int> t) {}
namespace Outer7 {
  namespace Space7 {
    void useit7d(Middle7::Thing7<int> t) {}
  }
}

%}

%template(Thing1Int) Thing1<int>;
%template(Thing2Int) Thing2<int>;
%template(Thing3Int) Thing3<int>;
%template(Thing4Int) Thing4<int>;
%template(Thing5Int) Thing5<int>;
%template(Thing7Int) Thing7<int>;


