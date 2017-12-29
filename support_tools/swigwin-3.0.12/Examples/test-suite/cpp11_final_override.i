// Test C++11 virtual specifier sequences (final and/or override on methods)
// Also check final/override - the two 'identifiers with special meaning' work as normal identifiers

%module cpp11_final_override

%warnfilter(SWIGWARN_PARSE_KEYWORD) final; // 'final' is a java keyword, renaming to '_final'
%warnfilter(SWIGWARN_PARSE_KEYWORD) override; // 'override' is a C# keyword, renaming to '_override'

%inline %{

struct Base {
  virtual void stuff() const {}
  virtual void override1() const {}
  virtual void override2() const {}
  virtual void finaloverride1() {}
  virtual void finaloverride2() {}
  virtual void finaloverride3() {}
  virtual void finaloverride4() const {}
  virtual ~Base() {}
};

struct Derived /*final*/ : Base {
  virtual void stuff() const noexcept override final {}
  virtual void override1() const noexcept override;
  virtual void override2() const noexcept override;
  virtual void final1() final {}
  virtual void final2() noexcept final {}
  virtual void final4() const final {}
  virtual void final5() const noexcept final {}
  virtual void finaloverride1() final override {}
  virtual void finaloverride2() override final {}
  virtual void finaloverride3() noexcept override final {}
  virtual void finaloverride4() const noexcept override final {}
  virtual ~Derived() override final {}
};
void Derived::override2() const noexcept {}

// Pure virtual methods
struct PureBase {
  virtual void pure1(int) const = 0;
  virtual void pure2(int) const = 0;
  virtual void pure3(int) const = 0;
  virtual void pure4(int) const = 0;
  virtual void pure5(int) const = 0;
  virtual ~PureBase() {}
};

struct PureDerived : PureBase {
  virtual void pure1(int) const override = 0;
  virtual void pure2(int) const final = 0;
  virtual void pure3(int) const override final = 0;
  virtual void pure4(int) const final override = 0;
  virtual void pure5(int) const noexcept final override = 0;
  virtual ~PureDerived() override final;
};
void PureDerived::pure1(int) const {}
void PureDerived::pure2(int) const {}
void PureDerived::pure3(int) const {}
void PureDerived::pure4(int) const {}
void PureDerived::pure5(int) const noexcept {}
PureDerived::~PureDerived() {}

// Destructors and virtual specifier sequences (final/override)
struct Destructors1 : Base {
  virtual ~Destructors1() override {}
};
struct Destructors2 : Base {
  virtual ~Destructors2() final {}
};
struct Destructors3 : Base {
  virtual ~Destructors3() noexcept final override {}
};
struct Destructors4 : Base {
  virtual ~Destructors4() noexcept override final {}
};

// Check the two 'identifiers with special meaning' work as normal identifiers
struct FinalOverrideMethods {
    virtual void final() {}
    virtual void override(int) {}
};
struct FinalOverrideVariables {
    int final;
    double override;
};
void final(int) {}
void override() {}
%}

%{
void Derived::override1() const noexcept {}
%}

// Example in documentation ... declarations only
%inline %{
struct BaseStruct {
  virtual void ab() const = 0;
  virtual void cd();
  virtual void ef();
  virtual ~BaseStruct();
};
struct DerivedStruct : BaseStruct {
  virtual void ab() const override;
  virtual void cd() final;
  virtual void ef() final override;
  virtual ~DerivedStruct() override;
};
struct DerivedNoVirtualStruct : BaseStruct {
  void ab() const override;
  void cd() final;
  void ef() final override;
  ~DerivedNoVirtualStruct() override;
};
%}

%{
void BaseStruct::cd() {}
void BaseStruct::ef() {}
BaseStruct::~BaseStruct() {}
void DerivedStruct::ab() const {}
void DerivedStruct::cd() {}
void DerivedStruct::ef() {}
DerivedStruct::~DerivedStruct() {}
void DerivedNoVirtualStruct::ab() const {}
void DerivedNoVirtualStruct::cd() {}
void DerivedNoVirtualStruct::ef() {}
DerivedNoVirtualStruct::~DerivedNoVirtualStruct() {}
%}
