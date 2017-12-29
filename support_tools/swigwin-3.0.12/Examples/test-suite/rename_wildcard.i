// Test rename overriding a wildcard rename
%module rename_wildcard

%rename(mm1) *::m1;
%rename(mm2) *::m2;
%rename(tt2) *::t2;
%rename(mm3) *::m3();
%rename(tt3) *::t3();
%rename(m_4) m4;
%rename(t_4) t4;
%rename(mm4) *::m4;
%rename(tt4) *::t4;
%rename(mm5) *::m5;
%rename(tt5) *::t5;
%rename(opint) *::operator int;
%rename(opdouble) *::operator double;

// No declaration
%rename(mm2a) GlobalWildStruct::m2;
%rename(mm2b) GlobalWildTemplateStruct::m2;
%rename(mm2c) Space::SpaceWildStruct::m2;
%rename(mm2d) Space::SpaceWildTemplateStruct::m2;
%rename(tt2b) GlobalWildTemplateStruct<int>::t2;
%rename(tt2d) Space::SpaceWildTemplateStruct<int>::t2;

// With declaration
%rename(mm3a) GlobalWildStruct::m3;
%rename(mm3b) GlobalWildTemplateStruct::m3;
%rename(mm3c) Space::SpaceWildStruct::m3;
%rename(mm3d) Space::SpaceWildTemplateStruct::m3;
%rename(tt3b) GlobalWildTemplateStruct<int>::t3;
%rename(tt3d) Space::SpaceWildTemplateStruct<int>::t3;

// Global override too
%rename(mm4a) GlobalWildStruct::m4;
%rename(mm4b) GlobalWildTemplateStruct::m4;
%rename(mm4c) Space::SpaceWildStruct::m4;
%rename(mm4d) Space::SpaceWildTemplateStruct::m4;
%rename(tt4b) GlobalWildTemplateStruct<int>::t4;
%rename(tt4d) Space::SpaceWildTemplateStruct<int>::t4;

// %extend renames
%extend GlobalWildStruct {
  %rename(mm5a) m5;
}
%extend GlobalWildTemplateStruct {
  %rename(mm5b) m5;
}
%extend GlobalWildTemplateStruct<int> {
  %rename(tt5b) t5;
}
namespace Space {
  %extend SpaceWildStruct {
    %rename(mm5c) m5;
  }
  %extend SpaceWildTemplateStruct {
    %rename(mm5d) m5;
  }
  %extend SpaceWildTemplateStruct<int> {
    %rename(tt5d) t5;
  }
}

// operators
%rename(opinta) GlobalWildStruct::operator int;
%rename(opintb) GlobalWildTemplateStruct::operator int;
%rename(opintc) Space::SpaceWildStruct::operator int;
%rename(opintd) Space::SpaceWildTemplateStruct::operator int;
%rename(opdoubleb) GlobalWildTemplateStruct<int>::operator double;
%rename(opdoubled) Space::SpaceWildTemplateStruct<int>::operator double;

%inline %{
struct GlobalWildStruct {
  void m1() {}
  void m2() {}
  void m3() {}
  void m4() {}
  void m5() {}
  operator int() { return 0; }
};
template<typename T> struct GlobalWildTemplateStruct {
  void m1() {}
  void m2() {}
  void t2() {}
  void m3() {}
  void t3() {}
  void m4() {}
  void t4() {}
  void m5() {}
  void t5() {}
  operator int() { return 0; }
  operator double() { return 0.0; }
};
namespace Space {
  struct SpaceWildStruct {
    void m1() {}
    void m2() {}
    void m3() {}
    void m4() {}
    void m5() {}
    operator int() { return 0; }
  };
  template<typename T> struct SpaceWildTemplateStruct {
    void m1() {}
    void m2() {}
    void t2() {}
    void m3() {}
    void t3() {}
    void m4() {}
    void t4() {}
    void m5() {}
    void t5() {}
    operator int() { return 0; }
    operator double() { return 0.0; }
  };
}

// Wild card renames expected for these
struct NoChangeStruct {
  void m1() {}
  void m2() {}
  void m3() {}
  void m4() {}
  void m5() {}
  operator int() { return 0; }
};
namespace Space {
  struct SpaceNoChangeStruct {
    void m1() {}
    void m2() {}
    void m3() {}
    void m4() {}
    void m5() {}
    operator int() { return 0; }
  };
}
%}

%template(GlobalWildTemplateStructInt) GlobalWildTemplateStruct<int>;
%template(SpaceWildTemplateStructInt) Space::SpaceWildTemplateStruct<int>;
