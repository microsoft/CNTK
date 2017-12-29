%module enum_ignore

// Similar to enum_missing C test, but with namespaces and using %ignore

%ignore N::C;

%inline %{
  namespace N {
    enum C { Red, Green, Blue };

    struct Draw {
      void DrawBW() {}
      void DrawC(C c) {}
      void DrawC_Ptr(C* c) {}
      void DrawC_ConstRef(C const& c) {}
    };
  }
%}


